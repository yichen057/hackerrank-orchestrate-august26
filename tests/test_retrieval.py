"""Retrieval invariants.

Runs against the committed dataset. No API calls.
"""
from __future__ import annotations

from retrieval import retrieve_candidates

LIMIT = 8


def _first_message_with_history(dataset):
    for row in dataset.messages:
        if dataset.history_by_user.get(row["user_id"]):
            return row
    raise AssertionError("no message in the dataset has user history")


def test_never_returns_more_than_the_limit(dataset):
    for row in dataset.messages:
        candidates = retrieve_candidates(dataset, row, row.get("message_text", ""))
        assert len(candidates) <= LIMIT


def test_is_deterministic_for_the_same_input(dataset):
    """The sort key is (-score, -timestamp, message_id), which is total, so the
    top-8 cannot depend on dictionary or file ordering."""
    row = _first_message_with_history(dataset)
    text = row.get("message_text", "")
    first = [item["message_id"] for item in retrieve_candidates(dataset, row, text)]
    second = [item["message_id"] for item in retrieve_candidates(dataset, row, text)]
    assert first == second


def test_candidates_are_distinct(dataset):
    for row in dataset.messages:
        ids = [item["message_id"] for item in retrieve_candidates(dataset, row, row.get("message_text", ""))]
        assert len(ids) == len(set(ids))


def test_candidates_are_ordered_by_descending_score(dataset):
    row = _first_message_with_history(dataset)
    scores = [
        item["relation"]["retrieval_score"]
        for item in retrieve_candidates(dataset, row, row.get("message_text", ""))
    ]
    assert scores == sorted(scores, reverse=True)


def test_candidates_are_strictly_older_than_the_incoming_message(dataset):
    """Nothing at or after the incoming timestamp may be used as evidence."""
    from datetime import datetime

    for row in dataset.messages:
        incoming_time = datetime.fromisoformat(row["created_at"])
        for item in retrieve_candidates(dataset, row, row.get("message_text", "")):
            assert datetime.fromisoformat(item["created_at"]) < incoming_time


def test_candidates_belong_to_the_same_user(dataset):
    """Evidence is drawn only from the recipient's own history, never another
    user's, which is what makes the same message route differently per user."""
    own_history = {
        user_id: {item["message_id"] for item in rows}
        for user_id, rows in dataset.history_by_user.items()
    }
    for row in dataset.messages:
        allowed = own_history.get(row["user_id"], set())
        for item in retrieve_candidates(dataset, row, row.get("message_text", "")):
            assert item["message_id"] in allowed


def test_a_message_with_no_history_yields_no_candidates(dataset):
    row = dict(_first_message_with_history(dataset))
    row["user_id"] = "user_does_not_exist"
    assert retrieve_candidates(dataset, row, row.get("message_text", "")) == []
