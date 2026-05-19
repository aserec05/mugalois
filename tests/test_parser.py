from mugalois.core.types import Triple
from mugalois.core.parser import json_to_triples
 
 
def test_parses_single_triple():
    raw = '{"triples":[{"s":"Mike","p":"st:isFriendWith","o":"Eleven"}]}'
    assert json_to_triples(raw) == {Triple("Mike", "st:isFriendWith", "Eleven")}
 
 
def test_parses_multiple_triples():
    """Multiple triples in the JSON must all be parsed correctly."""
    raw = '{"triples":[{"s":"Mike","p":"p","o":"Eleven"},{"s":"Dustin","p":"p","o":"Max"}]}'
    result = json_to_triples(raw)
    assert Triple("Mike", "p", "Eleven") in result
    assert Triple("Dustin", "p", "Max") in result
    assert len(result) == 2
 
 
def test_returns_set_no_duplicates():
    raw = '{"triples":[{"s":"Mike","p":"p","o":"Eleven"},{"s":"Mike","p":"p","o":"Eleven"}]}'
    assert len(json_to_triples(raw)) == 1
 
 
def test_empty_triples_list():
    assert json_to_triples('{"triples":[]}') == set()
 
 
def test_invalid_json_returns_empty():
    assert json_to_triples("this is not json") == set()
 
 
def test_missing_triples_key_returns_empty():
    assert json_to_triples('{"results":[{"s":"Mike","p":"p","o":"Eleven"}]}') == set()
 
 
def test_missing_s_field_returns_empty():
    assert json_to_triples('{"triples":[{"p":"p","o":"Eleven"}]}') == set()
 
 
def test_missing_o_field_returns_empty():
    assert json_to_triples('{"triples":[{"s":"Mike","p":"p"}]}') == set()
 
 
def test_empty_string_returns_empty():
    assert json_to_triples("") == set()
 
 
def test_null_json_returns_empty():
    assert json_to_triples("null") == set()
 
 
def test_llm_adds_markdown_fences_returns_empty():
    raw = '```json\n{"triples":[{"s":"Mike","p":"p","o":"Eleven"}]}\n```'
    assert json_to_triples(raw) == set()