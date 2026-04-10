"""Tests for ragevals.config -- RAGConfig and load_env."""

import os
from unittest.mock import patch

import pytest

from ragevals.config import RAGConfig, load_env


# ---------------------------------------------------------------------------
# RAGConfig defaults
# ---------------------------------------------------------------------------

class TestRAGConfigDefaults:
    def test_default_chunk_size(self):
        cfg = RAGConfig()
        assert cfg.chunk_size == 500

    def test_default_chunk_overlap(self):
        cfg = RAGConfig()
        assert cfg.chunk_overlap == 50

    def test_default_top_k(self):
        cfg = RAGConfig()
        assert cfg.top_k == 3

    def test_default_embedding_model(self):
        cfg = RAGConfig()
        assert cfg.embedding_model == "text-embedding-3-small"

    def test_default_generation_model(self):
        cfg = RAGConfig()
        assert cfg.generation_model == "gpt-4o-mini"

    def test_default_temperature(self):
        cfg = RAGConfig()
        assert cfg.temperature == 0.0

    def test_default_use_reranker(self):
        cfg = RAGConfig()
        assert cfg.use_reranker is False


# ---------------------------------------------------------------------------
# RAGConfig.name property
# ---------------------------------------------------------------------------

class TestRAGConfigName:
    def test_name_default(self):
        cfg = RAGConfig()
        assert cfg.name == "chunk500_top3_temp0.0"

    def test_name_custom(self):
        cfg = RAGConfig(chunk_size=256, top_k=5, temperature=0.3)
        assert cfg.name == "chunk256_top5_temp0.3"


# ---------------------------------------------------------------------------
# RAGConfig.from_dict
# ---------------------------------------------------------------------------

class TestRAGConfigFromDict:
    def test_from_dict_basic(self):
        d = {"chunk_size": 1024, "top_k": 10}
        cfg = RAGConfig.from_dict(d)
        assert cfg.chunk_size == 1024
        assert cfg.top_k == 10
        # Other fields should remain defaults
        assert cfg.temperature == 0.0

    def test_from_dict_ignores_unknown_keys(self):
        d = {"chunk_size": 256, "unknown_key": "should_be_ignored"}
        cfg = RAGConfig.from_dict(d)
        assert cfg.chunk_size == 256
        assert not hasattr(cfg, "unknown_key")


# ---------------------------------------------------------------------------
# load_env
# ---------------------------------------------------------------------------

class TestLoadEnv:
    @patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test-key-12345"}, clear=False)
    def test_load_env_returns_dict_with_api_key(self, tmp_path):
        # Create a temporary .env file
        env_file = tmp_path / ".env"
        env_file.write_text("OPENAI_API_KEY=sk-test-key-12345\n")
        result = load_env(str(env_file))
        assert isinstance(result, dict)
        assert "OPENAI_API_KEY" in result

    @patch.dict(os.environ, {}, clear=True)
    def test_load_env_raises_without_api_key(self, tmp_path):
        # Create an empty .env file
        env_file = tmp_path / ".env"
        env_file.write_text("")
        # Remove OPENAI_API_KEY if set
        os.environ.pop("OPENAI_API_KEY", None)
        with pytest.raises(EnvironmentError, match="OPENAI_API_KEY"):
            load_env(str(env_file))

    @patch.dict(
        os.environ,
        {"OPENAI_API_KEY": "sk-test", "COHERE_API_KEY": "co-test"},
        clear=False,
    )
    def test_load_env_includes_cohere_key(self, tmp_path):
        env_file = tmp_path / ".env"
        env_file.write_text(
            "OPENAI_API_KEY=sk-test\nCOHERE_API_KEY=co-test\n"
        )
        result = load_env(str(env_file))
        assert "COHERE_API_KEY" in result
