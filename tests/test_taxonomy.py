from signal_lag.analysis import taxonomy as tax_mod
from signal_lag.analysis.embeddings import Embedder
from signal_lag.config import Taxonomy, Topic


def _taxonomy():
    return Taxonomy(
        safety_topics=[
            Topic("interp", "Interpretability", "safety",
                  ["mechanistic interpretability circuits features"]),
        ],
        capability_topics=[
            Topic("agents", "Agents", "capability",
                  ["autonomous agents plan and act tool use"]),
        ],
        pairings=[],
        tag_threshold=0.05,
        max_tags_per_paper=2,
    )


def test_tagging_assigns_nearest_topic():
    tax = _taxonomy()
    # Force the deterministic offline backend.
    embedder = Embedder(model_name="does-not-exist")
    embedder.backend = "tfidf-svd"

    texts = [
        "We reverse-engineer circuits and interpretable features in transformers.",
        "An autonomous agent plans and uses tools to act over many steps.",
    ]
    ids = ["p_interp", "p_agent"]
    # Fit fallback on a corpus that includes seeds + papers so vocab overlaps.
    corpus = [s for t in tax.all_topics for s in t.seeds] + texts
    embedder.embed(corpus)  # fits the vectorizer
    vecs = embedder.embed(texts)

    centroids = tax_mod.build_topic_centroids(tax, embedder)
    rows = tax_mod.tag_papers(ids, vecs, centroids, tax)

    tags = {}
    for aid, key, score in rows:
        tags.setdefault(aid, []).append(key)
    # Top tag for each paper should be the matching topic.
    assert tags["p_interp"][0] == "interp"
    assert tags["p_agent"][0] == "agents"
