"""Tests for ragevals.datasets -- built-in data, I/O, validation."""

import json
import tempfile
from pathlib import Path

import pytest

from ragevals.datasets import (
    ACME_KNOWLEDGE_BASE,
    GOLDEN_TEST_CASES,
    load_dataset,
    save_dataset,
    validate_dataset,
)


# ---------------------------------------------------------------------------
# Built-in data
# ---------------------------------------------------------------------------

class TestAcmeKnowledgeBase:
    def test_length(self):
        assert len(ACME_KNOWLEDGE_BASE) == 15

    def test_has_required_keys(self):
        for doc in ACME_KNOWLEDGE_BASE:
            assert "id" in doc
            assert "title" in doc
            assert "content" in doc

    def test_ids_are_unique(self):
        ids = [doc["id"] for doc in ACME_KNOWLEDGE_BASE]
        assert len(ids) == len(set(ids))

    def test_content_is_nonempty(self):
        for doc in ACME_KNOWLEDGE_BASE:
            assert len(doc["content"].strip()) > 0


class TestGoldenTestCases:
    def test_length(self):
        assert len(GOLDEN_TEST_CASES) == 12

    def test_has_required_keys(self):
        for tc in GOLDEN_TEST_CASES:
            assert "query" in tc
            assert "reference" in tc

    def test_has_category(self):
        for tc in GOLDEN_TEST_CASES:
            assert "category" in tc
            assert len(tc["category"]) > 0

    def test_categories_are_known(self):
        known = {"returns", "shipping", "warranty", "support", "products", "payments", "loyalty"}
        for tc in GOLDEN_TEST_CASES:
            assert tc["category"] in known, f"Unknown category: {tc['category']}"


# ---------------------------------------------------------------------------
# Dataset I/O
# ---------------------------------------------------------------------------

class TestLoadSaveDataset:
    def test_round_trip(self, tmp_path):
        data = [{"query": "q1", "reference": "r1"}, {"query": "q2", "reference": "r2"}]
        path = tmp_path / "test_dataset.json"
        save_dataset(data, path)
        loaded = load_dataset(path)
        assert loaded == data

    def test_save_creates_parent_dirs(self, tmp_path):
        path = tmp_path / "nested" / "dir" / "data.json"
        save_dataset([{"a": 1}], path)
        assert path.exists()
        loaded = load_dataset(path)
        assert loaded == [{"a": 1}]

    def test_load_nonexistent_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_dataset(tmp_path / "does_not_exist.json")

    def test_save_preserves_unicode(self, tmp_path):
        data = [{"query": "Wie ist die Rueckgabepolitik?", "reference": "30 Tage."}]
        path = tmp_path / "unicode.json"
        save_dataset(data, path)
        loaded = load_dataset(path)
        assert loaded[0]["query"] == data[0]["query"]


# ---------------------------------------------------------------------------
# validate_dataset
# ---------------------------------------------------------------------------

class TestValidateDataset:
    def test_valid_dataset(self):
        dataset = [
            {"query": "What is X?", "reference": "X is Y."},
            {"query": "How does Z work?", "reference": "Z works by ..."},
        ]
        errors = validate_dataset(dataset)
        assert errors == []

    def test_missing_query(self):
        dataset = [{"reference": "Some answer."}]
        errors = validate_dataset(dataset)
        assert len(errors) == 1
        assert "query" in errors[0].lower()

    def test_empty_query(self):
        dataset = [{"query": "  ", "reference": "Answer."}]
        errors = validate_dataset(dataset)
        assert len(errors) == 1

    def test_missing_reference(self):
        dataset = [{"query": "Question?"}]
        errors = validate_dataset(dataset)
        assert len(errors) == 1
        assert "reference" in errors[0].lower()

    def test_not_a_dict(self):
        dataset = ["not a dict"]
        errors = validate_dataset(dataset)
        assert len(errors) == 1
        assert "not a dict" in errors[0].lower()

    def test_multiple_errors(self):
        dataset = [
            {"query": "", "reference": "OK"},
            {"query": "OK"},
            "bad",
        ]
        errors = validate_dataset(dataset)
        assert len(errors) == 3

    def test_golden_test_cases_are_valid(self):
        errors = validate_dataset(GOLDEN_TEST_CASES)
        assert errors == [], f"GOLDEN_TEST_CASES has errors: {errors}"
