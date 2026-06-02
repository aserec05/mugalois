"""
Triples Extraction from Wikidata, YAGO or DBpedia via SPARQL.


Example
------------------
    from ground_truth import GroundTruthExtractor

    gt = GroundTruthExtractor(source="wikidata")
    triples = gt.fetch(
        predicate="birthPlace",
        seeds=["Albert_Einstein", "Marie_Curie"],
    )
    for t in triples:
        print(t)
    # (http://www.wikidata.org/entity/Q937, birthPlace, Ulm)
"""

from __future__ import annotations

import logging
from typing import Literal

try:
    from SPARQLWrapper import SPARQLWrapper, JSON
except ImportError:
    raise ImportError("Installe SPARQLWrapper :  pip install SPARQLWrapper")

from mugalois.core.types import Triple

logger = logging.getLogger(__name__)

# ─── Consts ───────────────────────────────────────────────────────────────────

ENDPOINTS = {
    "yago":     "https://yago-knowledge.org/sparql/query",
    "dbpedia":  "https://dbpedia.org/sparql",
    "wikidata": "https://query.wikidata.org/sparql",
}

RESOURCE_PREFIXES = {
    "yago":    "https://yago-knowledge.org/resource/",
    "dbpedia": "http://dbpedia.org/resource/",
    # Wikidata uses QIDs — seeds are resolved via label lookup
}

PREDICATES = {
    "yago": {
        "birthPlace":  "https://schema.org/birthPlace",
        "deathPlace":  "https://schema.org/deathPlace",
        "spouse":      "https://schema.org/spouse",
        "nationality": "https://schema.org/nationality",
        "alumniOf":    "https://schema.org/alumniOf",
        "award":       "https://schema.org/award",
        "employer":    "https://schema.org/worksFor",
    },
    "dbpedia": {
        "birthPlace":  "http://dbpedia.org/ontology/birthPlace",
        "deathPlace":  "http://dbpedia.org/ontology/deathPlace",
        "spouse":      "http://dbpedia.org/ontology/spouse",
        "nationality": "http://dbpedia.org/ontology/nationality",
        "almaMater":   "http://dbpedia.org/ontology/almaMater",
        "award":       "http://dbpedia.org/ontology/award",
    },
    "wikidata": {
        "birthPlace":  "wdt:P19",
        "deathPlace":  "wdt:P20",
        "spouse":      "wdt:P26",
        "nationality": "wdt:P27",
        "almaMater":   "wdt:P69",
        "award":       "wdt:P166",
        "employer":    "wdt:P108",
    },
}

YAGO_TO_DBPEDIA_PREDICATE = {
    "birthPlace":  "birthPlace",
    "deathPlace":  "deathPlace",
    "spouse":      "spouse",
    "nationality": "nationality",
    "award":       "award",
}

DEFAULT_TIMEOUT = 30
DEFAULT_LIMIT   = 500


# ─── Main class ───────────────────────────────────────────────────────────────

class GroundTruthExtractor:
    """
    Retrieve RDF triples from Wikidata, YAGO or DBpedia.

    source : "wikidata" | "yago" | "dbpedia"
    timeout : int
    """

    def __init__(
        self,
        source: Literal["wikidata", "yago", "dbpedia"] = "wikidata",
        timeout: int = DEFAULT_TIMEOUT,
    ) -> None:
        if source not in ENDPOINTS:
            raise ValueError(
                f"Source inconnue '{source}'. Choix possibles : {list(ENDPOINTS)}"
            )
        self.source        = source
        self.endpoint_url  = ENDPOINTS[source]
        self.predicate_map = PREDICATES[source]

        self._sparql = SPARQLWrapper(self.endpoint_url)
        self._sparql.setReturnFormat(JSON)
        self._sparql.setTimeout(timeout)
        self._sparql.addCustomHttpHeader(
            "Accept", "application/sparql-results+json"
        )
        # Wikidata requires a User-Agent header
        if source == "wikidata":
            self._sparql.addCustomHttpHeader(
                "User-Agent", "muGalois-evaluation/1.0"
            )

    # ── Public API ────────────────────────────────────────────────────────────

    def fetch(
        self,
        predicate: str,
        seeds: list[str] | None = None,
        seed_side: Literal["subject", "object"] = "subject",
        limit: int = DEFAULT_LIMIT,
    ) -> set[Triple]:
        """
        Retrieve ground-truth triples for a given predicate.

        Return
        ------
        set[Triple]
        """
        if self.source == "wikidata":
            return self._fetch_wikidata(predicate, seeds, limit)

        predicate_uri = self._resolve_predicate(predicate)
        seed_uris     = self._build_seed_uris(seeds) if seeds else None
        query         = self._build_query(predicate_uri, seed_uris, seed_side, limit)

        try:
            raw = self._run_query(query)
        except RuntimeError as exc:
            if self.source == "yago" and "HTML" in str(exc):
                logger.warning("YAGO endpoint returned HTML — retrying on DBpedia.")
                return self._fetch_via_dbpedia(predicate, seeds, seed_side, limit)
            raise

        triples = self._parse(raw, predicate_uri)
        logger.info("%d triples — predicate '%s', source '%s'",
                    len(triples), predicate, self.source)
        return triples

    def available_predicates(self) -> list[str]:
        return list(self.predicate_map.keys())

    # ── Wikidata ──────────────────────────────────────────────────────────────

    def _fetch_wikidata(
        self,
        predicate: str,
        seeds: list[str] | None,
        limit: int,
    ) -> set[Triple]:
        """
        Wikidata-specific fetch.
        Seeds are plain entity names (e.g. "Albert_Einstein").
        The query resolves them by rdfs:label and returns the place label as object.
        Returns one triple per subject — one value per entity.
        """
        prop = self._resolve_predicate(predicate)

        if seeds:
            names = [s.replace("_", " ") for s in seeds]
            values_block = "VALUES ?label { " + " ".join(
                f'"{n}"@en' for n in names
            ) + " }"
            query = f"""
SELECT DISTINCT ?person ?personLabel ?placeLabel WHERE {{
  {values_block}
  ?person rdfs:label ?label .
  ?person {prop} ?place .
  ?place rdfs:label ?placeLabel .
  FILTER(LANG(?placeLabel) = "en")
}}
LIMIT {limit}
"""
        else:
            query = f"""
SELECT DISTINCT ?person ?placeLabel WHERE {{
  ?person {prop} ?place .
  ?place rdfs:label ?placeLabel .
  FILTER(LANG(?placeLabel) = "en")
}}
LIMIT {limit}
"""

        raw = self._run_query(query)
        return self._parse_wikidata(raw, predicate)

    def _parse_wikidata(self, raw: dict, predicate: str) -> set[Triple]:
        triples  = set()
        bindings = raw.get("results", {}).get("bindings", [])
        for b in bindings:
            s_val = b.get("person", {}).get("value", "")
            o_val = b.get("placeLabel", {}).get("value", "")
            if s_val and o_val:
                triples.add(Triple(s_val, predicate, o_val))
        return triples

    # ── YAGO / DBpedia ────────────────────────────────────────────────────────

    def _fetch_via_dbpedia(
        self,
        predicate: str,
        seeds: list[str] | None,
        seed_side: str,
        limit: int,
    ) -> set[Triple]:
        short_name = self._short_name(predicate)
        if short_name not in YAGO_TO_DBPEDIA_PREDICATE:
            raise RuntimeError(
                f"YAGO endpoint is unavailable and predicate '{predicate}' "
                f"has no DBpedia equivalent."
            )
        fallback = GroundTruthExtractor(source="dbpedia")
        return fallback.fetch(
            predicate=YAGO_TO_DBPEDIA_PREDICATE[short_name],
            seeds=seeds,
            seed_side=seed_side,
            limit=limit,
        )

    def _short_name(self, predicate: str) -> str:
        if not predicate.startswith("http"):
            return predicate
        for short, uri in self.predicate_map.items():
            if uri == predicate:
                return short
        return predicate

    def _resolve_predicate(self, predicate: str) -> str:
        if predicate.startswith("http") or predicate.startswith("wdt:"):
            return predicate
        if predicate not in self.predicate_map:
            raise ValueError(
                f"Prédicat inconnu '{predicate}' pour '{self.source}'.\n"
                f"Disponibles : {list(self.predicate_map)}"
            )
        return self.predicate_map[predicate]

    def _build_seed_uris(self, seeds: list[str]) -> list[str]:
        prefix = RESOURCE_PREFIXES.get(self.source, "")
        return [
            s if s.startswith("http") else prefix + s
            for s in seeds
        ]

    def _build_query(
        self,
        predicate_uri: str,
        seed_uris: list[str] | None,
        seed_side: str,
        limit: int,
    ) -> str:
        values_block = ""
        if seed_uris:
            uris_str = " ".join(f"<{u}>" for u in seed_uris)
            variable = "?s" if seed_side == "subject" else "?o"
            values_block = f"VALUES {variable} {{ {uris_str} }}"

        return (
            f"SELECT DISTINCT ?s ?o WHERE {{\n"
            f"  {values_block}\n"
            f"  ?s <{predicate_uri}> ?o .\n"
            f"}}\n"
            f"LIMIT {limit}"
        )

    # ── Query execution ───────────────────────────────────────────────────────

    def _run_query(self, query: str) -> dict:
        self._sparql.setQuery(query)
        try:
            raw = self._sparql.query().convert()
        except Exception as exc:
            raise RuntimeError(
                f"SPARQL query failed ({self.endpoint_url}) : {exc}"
            ) from exc

        if isinstance(raw, bytes):
            preview = raw[:200].decode("utf-8", errors="replace")
            raise RuntimeError(
                f"Endpoint '{self.endpoint_url}' returned HTML instead of JSON.\n"
                f"Response preview: {preview}"
            )
        return raw

    def _parse(self, raw: dict, predicate_uri: str) -> set[Triple]:
        triples  = set()
        bindings = raw.get("results", {}).get("bindings", [])
        for b in bindings:
            s_val = b.get("s", {}).get("value", "")
            o_val = b.get("o", {}).get("value", "")
            if s_val and o_val:
                triples.add(Triple(s_val, predicate_uri, o_val))
        return triples