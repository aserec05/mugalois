"""
Triples Extraction from YAGO or DBpedia via SPARQL.


Example
------------------
    from ground_truth import GroundTruthExtractor

    gt = GroundTruthExtractor(source="yago")
    triples = gt.fetch(
        predicate="birthPlace",
        seeds=["Albert_Einstein", "Marie_Curie"],
    )
    for t in triples:
        print(t)
    # {"s": "https://yago.../Albert_Einstein", "p": "https://schema.org/birthPlace", "o": "..."}

"""

from __future__ import annotations

import logging
from typing import Literal

try:
    from SPARQLWrapper import SPARQLWrapper, JSON
except ImportError:
    raise ImportError("Installe SPARQLWrapper :  pip install SPARQLWrapper")

logger = logging.getLogger(__name__)

# Consts

ENDPOINTS = {
    "yago":    "https://yago-knowledge.org/sparql/query",
    "dbpedia": "https://dbpedia.org/sparql",
}

RESOURCE_PREFIXES = {
    "yago":    "https://yago-knowledge.org/resource/",
    "dbpedia": "http://dbpedia.org/resource/",
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
}


YAGO_TO_DBPEDIA_PREDICATE = {
    "birthPlace":  "birthPlace",
    "deathPlace":  "deathPlace",
    "spouse":      "spouse",
    "nationality": "nationality",
    "award":       "award",
}

YAGO_TO_DBPEDIA_RESOURCE_PREFIX = {
    "yago":    "https://yago-knowledge.org/resource/",
    "dbpedia": "http://dbpedia.org/resource/",
}

DEFAULT_TIMEOUT = 30
DEFAULT_LIMIT   = 500


class GroundTruthExtractor:
    """
    Retrieve triplets RDF

    source : "yago" | "dbpedia"
    timeout : int   — timeout SPARQL en secondes
    """

    def __init__(
        self,
        source: Literal["yago", "dbpedia"] = "yago",
        timeout: int = DEFAULT_TIMEOUT,
    ) -> None:
        if source not in ENDPOINTS:
            raise ValueError(
                f"Source inconnue '{source}'. Choix possibles : {list(ENDPOINTS)}"
            )
        self.source          = source
        self.endpoint_url    = ENDPOINTS[source]
        self.predicate_map   = PREDICATES[source]
        self.resource_prefix = RESOURCE_PREFIXES[source]

        self._sparql = SPARQLWrapper(self.endpoint_url)
        self._sparql.setReturnFormat(JSON)
        self._sparql.setTimeout(timeout)


        self._sparql.addCustomHttpHeader(
            "Accept", "application/sparql-results+json"
        )

    # API publique

    def fetch(
        self,
        predicate: str,
        seeds: list[str] | None = None,
        seed_side: Literal["subject", "object"] = "subject",
        limit: int = DEFAULT_LIMIT,
    ) -> list[dict]:
        """
        Retrieve by a predicate given.

        Paramètres
        ----------
        predicate : str
            example ("birthPlace") ou full URI.
        seeds : list[str] | None
            ex. ["Albert_Einstein"].
           Can be None.
        seed_side : "subject" | "object"
        limit : int
            Nombre max de triplets

        Return
        --------
        list[dict]  — with keys "s", "p", "o"
        """
        predicate_uri = self._resolve_predicate(predicate)
        seed_uris     = self._build_seed_uris(seeds) if seeds else None
        query         = self._build_query(predicate_uri, seed_uris, seed_side, limit)

        logger.debug("Requête SPARQL :\n%s", query)

        try:
            raw = self._run_query(query)
        except RuntimeError as exc:
            
            if self.source == "yago" and "HTML" in str(exc):
                logger.warning(
                    "YAGO endpoint returned HTML — retrying on DBpedia fallback."
                )
                return self._fetch_via_dbpedia(predicate, seeds, seed_side, limit)
            raise

        triples = self._parse(raw, predicate_uri)

        logger.info(
            "%d triplets récupérés — prédicat '%s', source '%s'",
            len(triples), predicate, self.source,
        )
        return triples

    def available_predicates(self) -> list[str]:
        """Retourne les noms courts de prédicats disponibles pour cette source."""
        return list(self.predicate_map.keys())

    # Fallback

    def _fetch_via_dbpedia(
        self,
        predicate: str,
        seeds: list[str] | None,
        seed_side: str,
        limit: int,
    ) -> list[dict]:
        """
        Retry the fetch on DBpedia when YAGO is unavailable.
        Entity names are reused as-is (local names are identical in both KGs).
        """

        short_name = self._short_name(predicate)
        if short_name not in YAGO_TO_DBPEDIA_PREDICATE:
            raise RuntimeError(
                f"YAGO endpoint is unavailable and predicate '{predicate}' "
                f"has no DBpedia equivalent. Cannot fetch ground truth."
            )

        dbpedia_predicate = YAGO_TO_DBPEDIA_PREDICATE[short_name]
        fallback = GroundTruthExtractor(source="dbpedia")
        return fallback.fetch(
            predicate=dbpedia_predicate,
            seeds=seeds,
            seed_side=seed_side,
            limit=limit,
        )

    def _short_name(self, predicate: str) -> str:
        """Return the short name of a predicate (reverse lookup from URI if needed)."""
        if not predicate.startswith("http"):
            return predicate
        for short, uri in self.predicate_map.items():
            if uri == predicate:
                return short
        return predicate

    # Build

    def _resolve_predicate(self, predicate: str) -> str:
        if predicate.startswith("http"):
            return predicate
        if predicate not in self.predicate_map:
            raise ValueError(
                f"Prédicat inconnu '{predicate}' pour la source '{self.source}'.\n"
                f"Disponibles : {list(self.predicate_map)}"
            )
        return self.predicate_map[predicate]

    def _build_seed_uris(self, seeds: list[str]) -> list[str]:
        uris = []
        for s in seeds:
            uri = s if s.startswith("http") else self.resource_prefix + s
            uris.append(uri)
        return uris

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

    # parsing

    def _run_query(self, query: str) -> dict:
        self._sparql.setQuery(query)
        try:
            raw = self._sparql.query().convert()
        except Exception as exc:
            raise RuntimeError(
                f"La requête SPARQL a échoué ({self.endpoint_url}) : {exc}"
            ) from exc

    
        if isinstance(raw, bytes):
            preview = raw[:200].decode("utf-8", errors="replace")
            raise RuntimeError(
                f"Endpoint '{self.endpoint_url}' returned HTML instead of JSON.\n"
                f"Possible causes: wrong URL, endpoint down, rate limit.\n"
                f"Response preview: {preview}"
            )

        return raw

    def _parse(self, raw: dict, predicate_uri: str) -> list[dict]:
        triples  = []
        bindings = raw.get("results", {}).get("bindings", [])
        for b in bindings:
            s_val = b.get("s", {}).get("value", "")
            o_val = b.get("o", {}).get("value", "")
            if s_val and o_val:
                triples.append({"s": s_val, "p": predicate_uri, "o": o_val})
        return triples