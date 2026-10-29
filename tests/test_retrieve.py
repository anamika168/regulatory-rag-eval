import numpy as np
import pytest

from regrag.chunk import build_chunks
from regrag.retrieve import BM25Index, DenseIndex, Retriever, reciprocal_rank_fusion, should_abstain


@pytest.fixture
def chunks(pages):
    return build_chunks(pages, "structure", size=1200, overlap=200)


def test_bm25_prefers_matching_passage():
    index = BM25Index(["Handelsbuch täglich bewerten", "Anlagebuch vierteljährlich bewerten", "Compliance"])
    scores = index.scores("Wie oft ist das Anlagebuch zu bewerten?")
    assert int(np.argmax(scores)) == 1
    assert index.scores("Kryptowerte")[0] == 0.0


def test_dense_index_is_cosine_and_survives_save_load(tmp_path, embedder, chunks):
    vectors = embedder.encode_passages([c.embed_text for c in chunks])
    dense = DenseIndex(vectors)
    scores, ids = dense.search(vectors[:1], k=3)
    assert ids[0][0] == 0 and scores[0][0] == pytest.approx(1.0, abs=1e-5)
    dense.save(tmp_path / "x.index")
    loaded = DenseIndex.load(tmp_path / "x.index")
    np.testing.assert_allclose(loaded.vectors, dense.vectors)


@pytest.mark.parametrize("method", ["dense", "bm25", "hybrid"])
def test_retriever_finds_trading_book_passage(method, embedder, chunks):
    retriever = Retriever(chunks, method=method, embedder=embedder if method != "bm25" else None)
    hits = retriever.search("Positionen des Handelsbuches täglich bewerten", k=3)
    assert hits[0].chunk.section_id == "BTR 2.2"
    assert [h.rank for h in hits] == [1, 2, 3]
    if method == "bm25":
        assert hits[0].dense_score is None
    else:
        assert 0.0 < hits[0].dense_score <= 1.0 + 1e-6


def test_bm25_with_embedder_still_carries_dense_scores(embedder, chunks):
    retriever = Retriever(chunks, method="bm25", embedder=embedder)
    hits = retriever.search("Anlagebuch vierteljährlich", k=2)
    assert all(h.dense_score is not None for h in hits)


def test_retriever_requires_embedder_for_dense(chunks):
    with pytest.raises(ValueError):
        Retriever(chunks, method="dense")


def test_rrf_rewards_agreement():
    fused = reciprocal_rank_fusion([[1, 2, 3], [3, 1, 4]])
    assert fused[0][0] == 1          # rank 1 and rank 2
    assert {i for i, _ in fused} == {1, 2, 3, 4}


def test_threshold_abstention(embedder, chunks):
    retriever = Retriever(chunks, method="dense", embedder=embedder)
    assert not should_abstain(retriever.search("Positionen des Handelsbuches täglich bewerten"), 0.3)
    assert should_abstain(retriever.search("Kryptoverwahrung nach MiCAR"), 0.3)
