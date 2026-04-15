"""Tests des validators de normalisation LLM dans app/schemas.py."""
import pytest
from app.schemas import Classification, Intent


# ---------------------------------------------------------------------------
# Classification.urgency
# ---------------------------------------------------------------------------

def test_urgency_alias_high():
    c = Classification(category="x", urgency="high", confidence=0.8, reasoning="r", needs_due_date=False, tags=[])
    assert c.urgency == "haute"

def test_urgency_alias_urgent():
    c = Classification(category="x", urgency="urgent", confidence=0.8, reasoning="r", needs_due_date=False, tags=[])
    assert c.urgency == "critique"

def test_urgency_alias_low():
    c = Classification(category="x", urgency="low", confidence=0.8, reasoning="r", needs_due_date=False, tags=[])
    assert c.urgency == "basse"

def test_urgency_alias_medium():
    c = Classification(category="x", urgency="medium", confidence=0.8, reasoning="r", needs_due_date=False, tags=[])
    assert c.urgency == "normale"

def test_urgency_fr_passthrough():
    c = Classification(category="x", urgency="critique", confidence=0.8, reasoning="r", needs_due_date=False, tags=[])
    assert c.urgency == "critique"

def test_urgency_unknown_raises():
    with pytest.raises(Exception):
        Classification(category="x", urgency="blocker", confidence=0.8, reasoning="r", needs_due_date=False, tags=[])


# ---------------------------------------------------------------------------
# Classification.confidence
# ---------------------------------------------------------------------------

def test_confidence_string_high():
    c = Classification(category="x", urgency="normale", confidence="high", reasoning="r", needs_due_date=False, tags=[])
    assert c.confidence == 0.9

def test_confidence_string_medium():
    c = Classification(category="x", urgency="normale", confidence="medium", reasoning="r", needs_due_date=False, tags=[])
    assert c.confidence == 0.6

def test_confidence_string_low():
    c = Classification(category="x", urgency="normale", confidence="low", reasoning="r", needs_due_date=False, tags=[])
    assert c.confidence == 0.3

def test_confidence_numeric_string():
    c = Classification(category="x", urgency="normale", confidence="0.75", reasoning="r", needs_due_date=False, tags=[])
    assert c.confidence == 0.75

def test_confidence_float_passthrough():
    c = Classification(category="x", urgency="normale", confidence=0.85, reasoning="r", needs_due_date=False, tags=[])
    assert c.confidence == 0.85


# ---------------------------------------------------------------------------
# Intent.payload
# ---------------------------------------------------------------------------

def test_intent_payload_dict_passthrough():
    intent = Intent(kind="new_task", confidence=0.9, payload={"title": "foo"})
    assert intent.payload == {"title": "foo"}

def test_intent_payload_scalar_wrapped():
    intent = Intent(kind="new_task", confidence=0.9, payload="texte brut")
    assert intent.payload == {"raw": "texte brut"}

def test_intent_payload_int_wrapped():
    intent = Intent(kind="query", confidence=0.9, payload=42)
    assert intent.payload == {"raw": 42}
