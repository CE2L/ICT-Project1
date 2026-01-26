from typing import List, Optional
from pydantic import BaseModel, Field

class InterviewAnalysisRequest(BaseModel):
    transcripts: List[str] = Field(
        ..., min_length=1, description="List of candidate interview transcripts"
    )
    reference: str = Field(
        ..., min_length=1, description="Expert reference answer for comparison"
    )


class InterviewGenerationRequest(BaseModel):
    job_position: str=Field(
        default="Frontend Developer",
        description="Job position for interview generation",
    )
    num_candidates: int=Field(
        default=3, ge=1, le=10, description="Number of candidates to generate"
    )


class CandidateScore(BaseModel):
    candidate_number: int
    cosine_score: float
    rouge_score: float
    overall_score: float
    grade: str


class HireDecision(BaseModel):
    selected_candidate: int
    reason: str
    scores: List[CandidateScore]


class InterviewResponse(BaseModel):
    report: str
    score: float
    cosine_score: float
    rouge_score: float
    grade: str
    iterations: List[dict]
    transcripts: Optional[List[str]] = None
    reference: Optional[str] = None
    hire_decision: Optional[HireDecision] = None
    ai_provider: Optional[str] = None