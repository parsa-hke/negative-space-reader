"""engine/late_interaction.py: ColBERT token embeddings and MaxSim (plan §0.2 #8 step 1), on the real small model."""

import numpy as np
import pytest


@pytest.fixture(scope="module")
def colbert():
    import huggingface_hub.constants

    before = huggingface_hub.constants.HF_HUB_OFFLINE
    huggingface_hub.constants.HF_HUB_OFFLINE = True  # the model is cached; no network in tests
    from engine.late_interaction import ColBERT

    yield ColBERT("answerdotai/answerai-colbert-small-v1", revision="934fa8bb4ce2284f4c2baa232d81aca4d076fa5e",
                  device="cpu")
    huggingface_hub.constants.HF_HUB_OFFLINE = before


def test_encode_shapes_and_normalisation(colbert):
    q = colbert.encode(["We do not sell your personal information."], mode="query")[0]
    d = colbert.encode(["Cookies help us remember you.", "!!!"], mode="doc")
    assert q.shape[1] == 96 and np.allclose(np.linalg.norm(q, axis=1), 1, atol=1e-3)
    assert 2 <= q.shape[0] <= colbert.query_maxlen and d[1].shape[0] >= 1  # punctuation-only text keeps a token


def test_maxsim_ranks_paraphrase_above_unrelated(colbert):
    from engine.late_interaction import maxsim

    proto = colbert.encode(["The company does not sell personal information to third parties."], mode="query")[0]
    docs = colbert.encode(["We never sell your personal data to anyone.",
                           "Our office is open from nine to five on weekdays."], mode="doc")
    same, other = maxsim(proto, docs[0]), maxsim(proto, docs[1])
    assert same > other and -1 <= other <= same <= 1


def test_section_scores_take_the_best_sentence_and_prototype(colbert):
    from engine.late_interaction import section_scores

    protos = {"sells": colbert.encode(["The company does not sell personal information."], mode="query"),
              "hours": colbert.encode(["The office opening hours are stated."], mode="query")}
    sentences = colbert.encode(["We use cookies.", "We do not sell personal information."], mode="doc")
    s = section_scores(sentences, protos)
    assert s["sells"] > s["hours"] and set(s) == {"sells", "hours"}
