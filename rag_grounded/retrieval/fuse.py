# retrieval/fuse.py — RRF needs no score normalisation, which is the point.
# BM25 scores and cosine similarities live on incomparable scales; ranks don't.


def reciprocal_rank_fusion(rankings: list[list[str]], k: int = 60) -> list[str]:
    scores: dict[str, float] = {}
    for ranking in rankings:
        for position, chunk_id in enumerate(ranking):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + position + 1)
    return sorted(scores, key=scores.get, reverse=True)
