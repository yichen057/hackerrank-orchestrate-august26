"""The shipped CSV.

Checks the committed `submission/output.csv` against the committed dataset.
This is the structural verification the README reports, made executable: it
closes the one invariant the writer does not assert, that message ids are unique.
"""
from __future__ import annotations

import csv

import pytest

from data_io import OUTPUT_COLUMNS
from schemas import ACTIONS, MESSAGE_TYPES

MAX_EVIDENCE = 2


@pytest.fixture(scope="module")
def submission(repo_root):
    with (repo_root / "submission" / "output.csv").open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def test_header_matches_the_required_columns(repo_root):
    with (repo_root / "submission" / "output.csv").open(newline="", encoding="utf-8") as handle:
        assert next(csv.reader(handle)) == OUTPUT_COLUMNS


def test_one_row_per_input_message_and_no_duplicates(submission, dataset):
    ids = [row["message_id"] for row in submission]
    assert len(ids) == len(dataset.messages)
    assert len(ids) == len(set(ids)), "duplicate message_id in the submission"
    assert set(ids) == {row["message_id"] for row in dataset.messages}


def test_every_action_and_type_is_in_the_allowed_enum(submission):
    for row in submission:
        assert row["action"] in ACTIONS, row
        assert row["message_type"] in MESSAGE_TYPES, row


def test_confidence_is_a_probability(submission):
    for row in submission:
        value = float(row["confidence"])
        assert 0.0 <= value <= 1.0, row


def test_reason_is_present_and_bounded(submission):
    for row in submission:
        assert 1 <= len(row["reason"]) <= 200, row


def test_evidence_is_none_or_at_most_two_known_ids(submission, dataset):
    known = {
        item["message_id"]
        for rows in dataset.history_by_user.values()
        for item in rows
    }
    for row in submission:
        raw = row["evidence_message_ids"]
        if raw == "none":
            continue
        ids = raw.split(";")
        assert 1 <= len(ids) <= MAX_EVIDENCE, row
        assert len(ids) == len(set(ids)), row
        for message_id in ids:
            assert message_id in known, row


def test_no_row_matches_the_fallback_signature(submission):
    """The shipped output contains no row matching the complete fallback.

    The CSV alone cannot prove which code path produced a row, so this test
    deliberately checks the full deterministic signature rather than inferring
    provenance from only action, type, and confidence.
    """
    from fallback import conservative_fallback

    fallback = conservative_fallback()
    signature = [
        row
        for row in submission
        if row["action"] == fallback.action
        and row["message_type"] == fallback.message_type
        and row["reason"] == fallback.reason
        and float(row["confidence"]) == fallback.confidence
        and row["evidence_message_ids"] == "none"
    ]
    assert signature == []


def test_every_scam_or_spam_message_was_muted(submission):
    """The safety property the product cares about: nothing classified as a
    threat was allowed to interrupt the user."""
    for row in submission:
        if row["message_type"] in {"scam", "spam"}:
            assert row["action"] == "mute", row
