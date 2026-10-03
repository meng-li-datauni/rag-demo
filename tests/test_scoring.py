from evaluate import EVAL_CASES, contains_fact, score_case
from rag import NO_ANSWER


def test_contains_fact_accepts_alternatives():
    assert contains_fact("ga is march 14, 2027", "14 march 2027|march 14, 2027")
    assert not contains_fact("ga is 2027", "14 march 2027|march 14, 2027")


def test_correct_answer_passes():
    score = score_case(["a.pdf"], ["27"], "They get 27 days [a.pdf]", {"a.pdf", "b.pdf"})
    assert score == {"retrieval_ok": True, "answer_ok": True, "missing": []}


def test_missing_fact_and_wrong_source_fail():
    score = score_case(["a.pdf"], ["bellmark", "212"], "Bellmark ordered the most", {"b.pdf"})
    assert not score["retrieval_ok"]
    assert not score["answer_ok"]
    assert score["missing"] == ["212"]


def test_answerable_question_must_not_decline():
    assert not score_case(["a.pdf"], [], NO_ANSWER, {"a.pdf"})["answer_ok"]


def test_unanswerable_question_must_decline():
    assert score_case([], [], NO_ANSWER, set())["answer_ok"]
    assert not score_case([], [], "The CEO earns EUR 1M.", set())["answer_ok"]


def test_eval_cases_are_well_formed():
    for question, sources, facts in EVAL_CASES:
        assert question.endswith("?")
        assert bool(sources) == bool(facts), question
