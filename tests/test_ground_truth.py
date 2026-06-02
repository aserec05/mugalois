"""


- Unitary tests  : without network, Mocking
- Integrations tests :  SPARQL (nedds internet)

"""

import sys
import os
import pytest
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from mugalois.core.types import Triple
from evaluation.ground_truth import GroundTruthExtractor, ENDPOINTS, PREDICATES, RESOURCE_PREFIXES


# simulations
def make_sparql_response(bindings: list[dict]) -> dict:
    return {"results": {"bindings": bindings}}


def make_binding(s: str, o: str) -> dict:
    return {
        "s": {"type": "uri", "value": s},
        "o": {"type": "uri", "value": o},
    }


@pytest.fixture
def yago_extractor():
    with patch("evaluation.ground_truth.SPARQLWrapper") as MockSPARQL:
        mock_instance = MagicMock()
        MockSPARQL.return_value = mock_instance
        extractor = GroundTruthExtractor(source="yago")
        extractor._mock = mock_instance
        yield extractor


@pytest.fixture
def dbpedia_extractor():
    with patch("evaluation.ground_truth.SPARQLWrapper") as MockSPARQL:
        mock_instance = MagicMock()
        MockSPARQL.return_value = mock_instance
        extractor = GroundTruthExtractor(source="dbpedia")
        extractor._mock = mock_instance
        yield extractor


class TestInit:

    def test_yago_source_accepted(self, yago_extractor):
        assert yago_extractor.source == "yago"

    def test_dbpedia_source_accepted(self, dbpedia_extractor):
        assert dbpedia_extractor.source == "dbpedia"

    def test_unknown_source_raises(self):
        with pytest.raises(ValueError, match="Source inconnue"):
            with patch("evaluation.ground_truth.SPARQLWrapper"):
                GroundTruthExtractor(source="wikidata")

    def test_correct_endpoint_yago(self, yago_extractor):
        assert yago_extractor.endpoint_url == ENDPOINTS["yago"]

    def test_correct_endpoint_dbpedia(self, dbpedia_extractor):
        assert dbpedia_extractor.endpoint_url == ENDPOINTS["dbpedia"]

    def test_available_predicates_yago(self, yago_extractor):
        preds = yago_extractor.available_predicates()
        assert "birthPlace" in preds
        assert "spouse" in preds

    def test_available_predicates_dbpedia(self, dbpedia_extractor):
        preds = dbpedia_extractor.available_predicates()
        assert "birthPlace" in preds

    def test_accept_header_is_set(self, yago_extractor):
        yago_extractor._mock.addCustomHttpHeader.assert_called_once_with(
            "Accept", "application/sparql-results+json"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# UNITARY — _resolve_predicate
# ═══════════════════════════════════════════════════════════════════════════════

class TestResolvePredicate:

    def test_short_name_resolves(self, yago_extractor):
        uri = yago_extractor._resolve_predicate("birthPlace")
        assert uri == "https://schema.org/birthPlace"

    def test_full_uri_passthrough(self, yago_extractor):
        full = "https://schema.org/birthPlace"
        assert yago_extractor._resolve_predicate(full) == full

    def test_unknown_short_name_raises(self, yago_extractor):
        with pytest.raises(ValueError, match="Prédicat inconnu"):
            yago_extractor._resolve_predicate("nonExistentPredicate")

    def test_dbpedia_predicate_resolves(self, dbpedia_extractor):
        uri = dbpedia_extractor._resolve_predicate("birthPlace")
        assert uri == "http://dbpedia.org/ontology/birthPlace"


# ═══════════════════════════════════════════════════════════════════════════════
# UNITARY — _build_seed_uris
# ═══════════════════════════════════════════════════════════════════════════════

class TestBuildSeedUris:

    def test_plain_name_gets_prefix(self, yago_extractor):
        uris = yago_extractor._build_seed_uris(["Albert_Einstein"])
        assert uris == [RESOURCE_PREFIXES["yago"] + "Albert_Einstein"]

    def test_full_uri_unchanged(self, yago_extractor):
        full = "https://yago-knowledge.org/resource/Albert_Einstein"
        uris = yago_extractor._build_seed_uris([full])
        assert uris == [full]

    def test_multiple_seeds(self, yago_extractor):
        uris = yago_extractor._build_seed_uris(["Einstein", "Curie"])
        assert len(uris) == 2

    def test_empty_seeds(self, yago_extractor):
        assert yago_extractor._build_seed_uris([]) == []


# ═══════════════════════════════════════════════════════════════════════════════
# UNITARY  — _build_query
# ═══════════════════════════════════════════════════════════════════════════════

class TestBuildQuery:

    PREDICATE_URI = "https://schema.org/birthPlace"

    def test_no_seeds_no_values_block(self, yago_extractor):
        query = yago_extractor._build_query(self.PREDICATE_URI, None, "subject", 100)
        assert "VALUES" not in query
        assert "LIMIT 100" in query

    def test_subject_seeds_uses_s_variable(self, yago_extractor):
        uris = ["https://yago-knowledge.org/resource/Albert_Einstein"]
        query = yago_extractor._build_query(self.PREDICATE_URI, uris, "subject", 100)
        assert "VALUES ?s" in query

    def test_object_seeds_uses_o_variable(self, yago_extractor):
        uris = ["https://yago-knowledge.org/resource/Ulm"]
        query = yago_extractor._build_query(self.PREDICATE_URI, uris, "object", 100)
        assert "VALUES ?o" in query


# ═══════════════════════════════════════════════════════════════════════════════
# UNITARY — _parse  — now returns set[Triple]
# ═══════════════════════════════════════════════════════════════════════════════

class TestParse:

    PREDICATE_URI = "https://schema.org/birthPlace"

    def test_returns_set_of_triples(self, yago_extractor):
        raw = make_sparql_response([
            make_binding("https://yago.org/Einstein", "https://yago.org/Ulm")
        ])
        result = yago_extractor._parse(raw, self.PREDICATE_URI)
        assert isinstance(result, set)
        assert all(isinstance(t, Triple) for t in result)

    def test_normal_binding(self, yago_extractor):
        raw = make_sparql_response([
            make_binding("https://yago.org/Albert_Einstein", "https://yago.org/Ulm")
        ])
        triples = yago_extractor._parse(raw, self.PREDICATE_URI)
        assert len(triples) == 1
        t = list(triples)[0]
        assert t.s == "https://yago.org/Albert_Einstein"
        assert t.p == self.PREDICATE_URI
        assert t.o == "https://yago.org/Ulm"

    def test_empty_bindings(self, yago_extractor):
        raw = make_sparql_response([])
        assert yago_extractor._parse(raw, self.PREDICATE_URI) == set()

    def test_missing_s_skipped(self, yago_extractor):
        raw = make_sparql_response([{"o": {"type": "uri", "value": "https://yago.org/Ulm"}}])
        assert yago_extractor._parse(raw, self.PREDICATE_URI) == set()

    def test_multiple_bindings(self, yago_extractor):
        raw = make_sparql_response([
            make_binding("https://yago.org/Einstein", "https://yago.org/Ulm"),
            make_binding("https://yago.org/Curie",   "https://yago.org/Warsaw"),
        ])
        assert len(yago_extractor._parse(raw, self.PREDICATE_URI)) == 2


# ═══════════════════════════════════════════════════════════════════════════════
# UNITARY — _run_query (HTML detection)
# ═══════════════════════════════════════════════════════════════════════════════

class TestRunQuery:

    def test_html_response_raises_runtime_error(self, yago_extractor):
        html_bytes = b"<!DOCTYPE html><html><head><title>YAGO</title></head></html>"
        yago_extractor._mock.query.return_value.convert.return_value = html_bytes
        with pytest.raises(RuntimeError, match="HTML"):
            yago_extractor._run_query("SELECT ?s WHERE { ?s ?p ?o } LIMIT 1")

    def test_json_response_passes_through(self, yago_extractor):
        json_response = make_sparql_response([
            make_binding("https://yago.org/Einstein", "https://yago.org/Ulm")
        ])
        yago_extractor._mock.query.return_value.convert.return_value = json_response
        result = yago_extractor._run_query("SELECT ?s ?o WHERE { ?s ?p ?o } LIMIT 1")
        assert result == json_response

    def test_sparql_exception_raises_runtime_error(self, yago_extractor):
        yago_extractor._mock.query.side_effect = Exception("Timeout")
        with pytest.raises(RuntimeError, match="SPARQL"):
            yago_extractor._run_query("SELECT ?s WHERE { ?s ?p ?o } LIMIT 1")


# ═══════════════════════════════════════════════════════════════════════════════
# UNITARY — fetch
# ═══════════════════════════════════════════════════════════════════════════════

class TestFetch:

    def _setup_mock(self, extractor, bindings):
        extractor._mock.query.return_value.convert.return_value = (
            make_sparql_response(bindings)
        )

    def test_fetch_returns_set_of_triples(self, yago_extractor):
        self._setup_mock(yago_extractor, [
            make_binding(
                "https://yago-knowledge.org/resource/Albert_Einstein",
                "https://yago-knowledge.org/resource/Ulm",
            )
        ])
        result = yago_extractor.fetch("birthPlace", seeds=["Albert_Einstein"])
        assert isinstance(result, set)
        assert all(isinstance(t, Triple) for t in result)

    def test_fetch_html_response_raises_runtime(self, yago_extractor):
        yago_extractor._mock.query.return_value.convert.return_value = b"<!DOCTYPE html>"
        with pytest.raises(RuntimeError, match="HTML"):
            yago_extractor.fetch("birthPlace", seeds=["Albert_Einstein"])

    def test_fetch_sparql_error_raises_runtime(self, yago_extractor):
        yago_extractor._mock.query.side_effect = Exception("Connexion refusée")
        with pytest.raises(RuntimeError, match="SPARQL"):
            yago_extractor.fetch("birthPlace", seeds=["Einstein"])

    def test_fetch_unknown_predicate_raises_value_error(self, yago_extractor):
        with pytest.raises(ValueError, match="Prédicat inconnu"):
            yago_extractor.fetch("invented_predicate", seeds=["Einstein"])

    def test_fetch_object_side_seeds(self, yago_extractor):
        self._setup_mock(yago_extractor, [
            make_binding(
                "https://yago-knowledge.org/resource/Albert_Einstein",
                "https://yago-knowledge.org/resource/Ulm",
            )
        ])
        yago_extractor.fetch("birthPlace", seeds=["Ulm"], seed_side="object")
        call_args = yago_extractor._mock.setQuery.call_args[0][0]
        assert "VALUES ?o" in call_args


# ═══════════════════════════════════════════════════════════════════════════════
# UNITARY — DBpedia fallback
# ═══════════════════════════════════════════════════════════════════════════════

class TestDBpediaFallback:

    def test_yago_html_triggers_dbpedia_fallback(self, yago_extractor):
        yago_extractor._mock.query.return_value.convert.return_value = b"<!DOCTYPE html>"

        with patch("evaluation.ground_truth.GroundTruthExtractor") as MockFallback:
            mock_instance = MagicMock()
            mock_instance.fetch.return_value = {
                Triple("http://dbpedia.org/resource/Albert_Einstein",
                       "http://dbpedia.org/ontology/birthPlace",
                       "http://dbpedia.org/resource/Ulm")
            }
            MockFallback.return_value = mock_instance
            result = yago_extractor.fetch("birthPlace", seeds=["Albert_Einstein"])

        assert len(result) == 1

    def test_yago_html_unknown_predicate_raises(self, yago_extractor):
        yago_extractor._mock.query.return_value.convert.return_value = b"<!DOCTYPE html>"
        with pytest.raises(RuntimeError, match="no DBpedia equivalent"):
            yago_extractor.fetch("employer", seeds=["Albert_Einstein"])

    def test_dbpedia_html_does_not_fallback(self, dbpedia_extractor):
        dbpedia_extractor._mock.query.return_value.convert.return_value = b"<!DOCTYPE html>"
        with pytest.raises(RuntimeError, match="HTML"):
            dbpedia_extractor.fetch("birthPlace", seeds=["Albert_Einstein"])


# ═══════════════════════════════════════════════════════════════════════════════
# INTEGRATION —  SPARQL calls
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.integration
class TestIntegrationYAGO:
    """pytest -v -m integration"""

    @pytest.fixture(scope="class")
    def extractor(self):
        return GroundTruthExtractor(source="yago", timeout=30)

    def test_fetch_einstein_birthplace(self, extractor):
        triples = extractor.fetch("birthPlace", seeds=["Albert_Einstein"])
        assert len(triples) >= 1
        assert all(isinstance(t, Triple) for t in triples)

    def test_fetch_multiple_seeds(self, extractor):
        triples = extractor.fetch("birthPlace", seeds=["Albert_Einstein", "Marie_Curie"])
        assert len(triples) >= 1


@pytest.mark.integration
class TestIntegrationDBpedia:

    @pytest.fixture(scope="class")
    def extractor(self):
        return GroundTruthExtractor(source="dbpedia", timeout=30)

    def test_fetch_einstein_birthplace_dbpedia(self, extractor):
        triples = extractor.fetch("birthPlace", seeds=["Albert_Einstein"])
        assert len(triples) >= 1
        assert all(isinstance(t, Triple) for t in triples)

    def test_result_structure(self, extractor):
        triples = extractor.fetch("birthPlace", seeds=["Albert_Einstein"])
        for t in triples:
            assert isinstance(t, Triple)
            assert t.p == "http://dbpedia.org/ontology/birthPlace"