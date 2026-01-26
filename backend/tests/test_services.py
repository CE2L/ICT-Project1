from unittest.mock import patch
import pytest


def test_calculate_grade_returns_a_for_high_score(evaluation_service):
    # GIVEN: a high evaluation score
    score = 0.9

    # WHEN: calculating the grade
    grade = evaluation_service.calculate_grade(score)

    # THEN: grade should be "A"
    assert grade == "A"


def test_calculate_grade_returns_d_for_low_score(evaluation_service):
    # GIVEN: a low evaluation score
    score = 0.3

    # WHEN: calculating the grade
    grade = evaluation_service.calculate_grade(score)

    # THEN: grade should be "D"
    assert grade == "D"


def test_cosine_similarity_returns_zero_for_orthogonal_vectors(evaluation_service):
    # GIVEN: two orthogonal vectors
    vec_a = [1, 0]
    vec_b = [0, 1]

    # WHEN: computing cosine similarity
    result = evaluation_service.cosine_sim(vec_a, vec_b)

    # THEN: similarity should be zero
    assert result == 0.0


def test_cosine_similarity_returns_one_for_identical_vectors(evaluation_service):
    # GIVEN: two identical vectors
    vec_a = [1, 1]
    vec_b = [1, 1]

    # WHEN: computing cosine similarity
    result = evaluation_service.cosine_sim(vec_a, vec_b)

    # THEN: similarity should be one
    assert result == pytest.approx(1.0)


def test_calculate_rouge_returns_zero_for_no_overlap(evaluation_service):
    # GIVEN: two texts with no overlapping tokens
    text1 = "hello world"
    text2 = "foo bar"

    # WHEN: calculating ROUGE score
    score = evaluation_service.calculate_rouge(text1, text2)

    # THEN: score should be zero
    assert score == 0.0


def test_calculate_rouge_returns_one_for_identical_text(evaluation_service):
    # GIVEN: two identical texts
    text1 = "hello world"
    text2 = "hello world"

    # WHEN: calculating ROUGE score
    score = evaluation_service.calculate_rouge(text1, text2)

    # THEN: score should be one
    assert score == 1.0


@patch("services.OpenAIService.get_embedding")
@patch("services.OpenAIService.fetch_chat_completion")
def test_get_hire_decision_selects_best_candidate(
    mock_chat_completion, mock_get_embedding, evaluation_service
):
    # GIVEN:
    # - mocked embeddings to control similarity results
    # - mocked chat completion to avoid external API calls
    mock_get_embedding.side_effect = [
        [0.9, 0.9], [0.5, 0.5],  # candidate 1 vs ref
        [0.3, 0.3], [0.5, 0.5],  # candidate 2 vs ref
        [0.1, 0.1], [0.5, 0.5],  # candidate 3 vs ref
    ]

    mock_chat_completion.return_value = "Candidate 1 matched criteria best"

    transcripts = [
        "Excellent answer",
        "Average answer",
        "Poor answer",
    ]
    reference = "Expert answer"

    # WHEN: evaluating hiring decision
    decision = evaluation_service.get_hire_decision(transcripts, reference)

    # THEN:
    # - candidate 1 should be selected
    # - scores should be generated for all candidates
    # - explanation should reference candidate 1
    assert decision.selected_candidate == 1
    assert len(decision.scores) == 3
    assert "Candidate 1" in decision.reason