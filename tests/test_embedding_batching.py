"""Tests for token-aware batch splitting in OpenAIEmbeddingFunction."""

from unittest.mock import MagicMock, call

from zotero_mcp.chroma_client import OpenAIEmbeddingFunction


class FakeOpenAIEmbeddingFunction(OpenAIEmbeddingFunction):
    """Subclass that skips real API client init."""

    def __init__(self, max_batch_tokens: int = 300000):
        # Skip parent __init__ — no API key needed
        self.model_name = "text-embedding-3-small"
        self.api_key = "fake"
        self.base_url = None
        self.client = MagicMock()
        self.max_batch_tokens = max_batch_tokens



def _make_mock_response(n):
    """Create a mock OpenAI embeddings response with n embeddings."""
    mock_response = MagicMock()
    mock_data = []
    for i in range(n):
        item = MagicMock()
        item.embedding = [0.1] * 3  # tiny fake embedding
        mock_data.append(item)
    mock_response.data = mock_data
    return mock_response


def test_small_batch_single_api_call():
    """Small batch that fits in one request should make exactly one API call."""
    ef = FakeOpenAIEmbeddingFunction(max_batch_tokens=300000)
    # 5 short docs, well under 300k tokens
    docs = ["short text"] * 5
    ef.client.embeddings.create.return_value = _make_mock_response(5)

    result = ef(docs)

    assert len(result) == 5
    assert ef.client.embeddings.create.call_count == 1


def test_large_batch_splits_into_multiple_api_calls():
    """Batch exceeding max_batch_tokens should be split into multiple API calls."""
    # Set a very low limit to force splitting
    ef = FakeOpenAIEmbeddingFunction(max_batch_tokens=100)
    # Each doc is ~50 chars ≈ ~12 tokens. Two docs ≈ 24 tokens > 100? No.
    # Use longer docs: 400 chars each ≈ 100 tokens each. 3 docs = 300 tokens > 100.
    docs = ["x" * 400] * 3
    ef.client.embeddings.create.side_effect = [
        _make_mock_response(1),
        _make_mock_response(1),
        _make_mock_response(1),
    ]

    result = ef(docs)

    assert len(result) == 3
    assert ef.client.embeddings.create.call_count == 3


def test_batch_splitting_preserves_order():
    """Embeddings should be returned in the same order as input documents."""
    ef = FakeOpenAIEmbeddingFunction(max_batch_tokens=100)
    docs = ["x" * 400, "y" * 400, "z" * 400]

    # Each call returns a distinct embedding so we can verify order
    def mock_create(model, input):
        resp = MagicMock()
        data = []
        for text in input:
            item = MagicMock()
            item.embedding = [ord(text[0])]  # 120 for 'x', 121 for 'y', 122 for 'z'
            data.append(item)
        resp.data = data
        return resp

    ef.client.embeddings.create.side_effect = mock_create

    result = ef(docs)

    assert result == [[120], [121], [122]]


def test_empty_input_returns_empty():
    """Empty input should return empty list without API calls."""
    ef = FakeOpenAIEmbeddingFunction()

    # Call the internal method directly — ChromaDB's wrapper rejects empty lists
    result = ef._embed_with_batching([])

    assert result == []
    assert ef.client.embeddings.create.call_count == 0


def test_single_large_doc_gets_own_batch():
    """A single doc that nearly fills the token limit should still be sent."""
    ef = FakeOpenAIEmbeddingFunction(max_batch_tokens=300000)
    # One very large doc
    docs = ["x" * 1000000]
    ef.client.embeddings.create.return_value = _make_mock_response(1)

    result = ef(docs)

    assert len(result) == 1
    assert ef.client.embeddings.create.call_count == 1
