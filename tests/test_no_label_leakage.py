"""Leave-one-out evaluation cannot see the answer it is being scored against.

The thirty solved samples are also the few-shot examples, so the evaluation has
to prove the held-out row's labels never reach the prompt. The guarantee comes
from *whitelist projection* onto the unlabelled input schema, not from deleting
keys. These tests pin that.
"""
from __future__ import annotations

import ast
from pathlib import Path

HARNESS = Path(__file__).resolve().parents[1] / "code" / "evaluation" / "main.py"


def _label_fields() -> set[str]:
    """Read LABEL_FIELDS out of the harness without importing it.

    Importing the harness would pull in the media stack (PyAV, Pillow) and the
    provider, which these tests deliberately do not need. Parsing the literal
    keeps the suite dependency-free while still failing if the harness changes.
    """
    tree = ast.parse(HARNESS.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "LABEL_FIELDS" for t in node.targets
        ):
            return set(ast.literal_eval(node.value))
    raise AssertionError("LABEL_FIELDS not found in the evaluation harness")


LABEL_FIELDS = _label_fields()


def test_label_fields_is_the_full_set_of_scored_columns():
    assert LABEL_FIELDS == {
        "action",
        "message_type",
        "reason",
        "confidence",
        "evidence_message_ids",
    }


def test_the_unlabelled_input_schema_carries_no_label_columns(dataset):
    assert not LABEL_FIELDS & set(dataset.messages[0].keys())


def test_the_solved_samples_do_carry_labels(dataset):
    """Guards the guard: if the samples ever stopped carrying labels, the test
    above would pass for the wrong reason."""
    assert LABEL_FIELDS <= set(dataset.samples[0].keys())


def test_projection_onto_the_input_schema_drops_every_label(dataset):
    input_fields = tuple(dataset.messages[0].keys())
    held_out = dataset.samples[0]
    incoming = {field: held_out[field] for field in input_fields}
    assert not LABEL_FIELDS & set(incoming)


def test_holding_one_out_leaves_exactly_twenty_nine_examples(dataset):
    assert len(dataset.samples) == 30
    for held_out in dataset.samples:
        others = [
            s for s in dataset.samples if s["message_id"] != held_out["message_id"]
        ]
        assert len(others) == 29, held_out["message_id"]


def test_sample_ids_are_unique(dataset):
    """The exclusion filters on message_id equality, so a duplicate id would
    silently leave the answer in the prompt."""
    ids = [s["message_id"] for s in dataset.samples]
    assert len(ids) == len(set(ids))


def test_solved_samples_never_appear_among_the_predicted_messages(dataset):
    """Two files, two id namespaces. The thirty labelled rows are not part of
    the hundred and ten being predicted."""
    predicted = {row["message_id"] for row in dataset.messages}
    solved = {row["message_id"] for row in dataset.samples}
    assert predicted & solved == set()
