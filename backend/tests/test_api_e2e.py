from unittest.mock import patch


def test_analyze_interviews_e2e(client):
    # GIVEN: Valid analysis payload
    payload = {
        "transcripts": ["Candidate text"],
        "reference": "Expert reference"
    }

    # WHEN: POST request is sent to /interviews/analyses
    response = client.post("/interviews/analyses", json=payload)

    # THEN: Status code 200 is returned with evaluation result
    if response.status_code != 200:
        print(f"Response status: {response.status_code}")
        print(f"Response body: {response.json()}")
    
    assert response.status_code == 200
    assert "hire_decision" in response.json()


def test_generate_interviews_endpoint_e2e(client):
    # GIVEN: Mocking interview generation service
    with patch("services.InterviewService.generate_content") as mock_gen:
        mock_gen.return_value = {
            "transcripts": ["Generated Answer 1"],
            "reference": "Generated Reference"
        }
        payload = {"job_position": "Data Scientist"}

        # WHEN: POST request is sent to /interviews/generations
        response = client.post("/interviews/generations", json=payload)

        # THEN: Status code 201 is returned
        if response.status_code != 201:
            print(f"Response status: {response.status_code}")
            print(f"Response body: {response.json()}")
        
        assert response.status_code == 201


def test_analyze_interviews_validation_error(client):
    # GIVEN: An invalid payload with empty transcripts
    # Pydantic field validation (min_length=1) will catch this
    payload = {"transcripts": [], "reference": "Expert"}

    # WHEN: POST request is sent to /interviews/analyses
    response = client.post("/interviews/analyses", json=payload)

    # THEN: FastAPI/Pydantic returns 422 (Unprocessable Entity) for schema validation errors
    assert response.status_code == 422