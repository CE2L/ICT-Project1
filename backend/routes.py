from fastapi import APIRouter, HTTPException, status, Query
from openai import OpenAIError

from models import (
    InterviewAnalysisRequest,
    InterviewGenerationRequest,
    InterviewResponse,
)
from services import (
    EvaluationService,
    InterviewService,
    OpenAIService,
    FriendliService,
    GeminiService
)

router = APIRouter(prefix="/interviews", tags=["Interviews"])

AI_PROVIDERS = {
    "openai": OpenAIService(),
    "friendli": FriendliService(),
    "gemini": GeminiService()
}


def get_services(provider_name: str):
    provider = AI_PROVIDERS.get(provider_name.lower(), AI_PROVIDERS["friendli"])
    return (
        InterviewService(ai_service=provider),
        EvaluationService(ai_service=provider)
    )


@router.post(
    "/generations",
    response_model=InterviewResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Generate mock interview data and run evaluation",
    description="""
Creates a complete, demo-ready interview dataset for a given job position and immediately evaluates it.

What this endpoint does:
- Uses the LLM to generate multiple candidate answers (default: 3) for the specified role
- Generates one expert reference answer that represents an ideal response
- Produces a cross-candidate analysis report (trends, differences, recommendations)
- Calculates quantitative similarity metrics against the reference:
  - Cosine similarity using embeddings
  - ROUGE-like overlap score
- Produces an overall score and grade
- Selects the best candidate and generates a short hiring rationale

Use cases:
- Live demo for reviewers without requiring manual input
- Quick regression check that the full workflow still runs end-to-end
- Generating sample data for UI screenshots and README documentation

Notes:
- This endpoint calls external AI services and may be slower than purely local endpoints.
- If the selected provider API key is missing/invalid, the request will fail.
""",
    responses={
        201: {
            "description": "Interview data generated successfully and analysis completed.",
        },
        422: {
            "description": "Validation error or unexpected response shape from AI generation step.",
        },
        502: {
            "description": "Upstream AI provider error while generating content or analysis.",
        },
    },
)
def create_interview_data(
    req: InterviewGenerationRequest,
    provider: str = Query("openai", description="AI engine to use: openai, friendli, gemini")
):
    int_service, eval_service = get_services(provider)
    try:
        generated_data = int_service.generate_content(req.job_position)

        return eval_service.process_analysis(
            transcripts=generated_data["transcripts"],
            reference=generated_data["reference"],
            position=req.job_position,
        )
    except KeyError as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Unexpected AI response format (missing key): {str(e)}",
        )
    except (OpenAIError, Exception) as e:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Error from {provider} during generation/analysis: {str(e)}",
        )


@router.post(
    "/analyses",
    response_model=InterviewResponse,
    summary="Analyze user-provided interview transcripts against a reference",
    description="""
Evaluates candidate interview answers provided by the user by comparing them to an expert reference answer.

What this endpoint does:
- Accepts 1+ candidate transcripts and a required expert reference answer
- Generates a structured cross-analysis report:
  - Common trends across candidates
  - Key differences and strengths/weaknesses
  - Actionable recommendations
- Computes quantitative metrics for the generated report vs the reference:
  - Cosine similarity (embeddings)
  - ROUGE-like overlap score
- Calculates an overall score and grade
- Scores each candidate individually and selects the best candidate
- Produces a concise hiring rationale for the selected candidate

Use cases:
- Real interview preparation: compare multiple answers to a target reference
- Measuring how close an LLM-generated report aligns with a ground-truth answer
- Demonstrating explainable scoring with both numeric metrics and qualitative output

Notes:
- At least one transcript is required.
- The LLM is used for report generation and hiring rationale, so latency/cost applies.
""",
    responses={
        200: {
            "description": "Analysis completed successfully.",
        },
        400: {
            "description": "Bad request (e.g., missing transcripts).",
        },
        502: {
            "description": "Upstream AI provider error while generating the report or rationale.",
        },
    },
)
def analyze_interviews(
    req: InterviewAnalysisRequest,
    provider: str = Query("openai", description="AI engine to use: openai, friendli, gemini")
):
    if not req.transcripts:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="At least one interview transcript is required.",
        )

    _, eval_service = get_services(provider)
    try:
        return eval_service.process_analysis(
            transcripts=req.transcripts,
            reference=req.reference,
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    except (OpenAIError, Exception) as e:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Error from {provider} during analysis: {str(e)}",
        )