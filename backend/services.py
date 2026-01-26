import os
import re
from abc import ABC, abstractmethod
from typing import List
from openai import OpenAI
from models import CandidateScore, HireDecision

class BaseAIService(ABC):
    @abstractmethod
    def fetch_chat_completion(self, prompt: str) -> str:
        pass

    @abstractmethod
    def get_embedding(self, text: str) -> List[float]:
        pass

    @abstractmethod
    def get_provider_name(self) -> str:
        pass

class OpenAIService(BaseAIService):
    def __init__(self):
        api_key = os.environ.get("OPENAI_API_KEY") or "DUMMY_FOR_TESTS"
        self.client = OpenAI(api_key=api_key)

    def fetch_chat_completion(self, prompt: str) -> str:
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "system", "content": "You are a helpful assistant that outputs only plain text or structured data as requested."},
                      {"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content

    def get_embedding(self, text: str) -> List[float]:
        response = self.client.embeddings.create(
            model="text-embedding-3-small",
            input=text,
        )
        return response.data[0].embedding

    def get_provider_name(self) -> str:
        return "OpenAI (gpt-4o-mini)"

class FriendliService(BaseAIService):
    def __init__(self):
        api_key = os.environ.get("FRIENDLI_API_KEY")
        self.has_key = bool(api_key)
        if self.has_key:
            self.client = OpenAI(
                api_key=api_key,
                base_url="https://inference.friendli.ai/v1"
            )

    def fetch_chat_completion(self, prompt: str) -> str:
        if not self.has_key:
            raise ValueError("Friendli API key not configured")

        response = self.client.chat.completions.create(
            model="meta-llama-3.1-8b-instruct",
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content

    def get_embedding(self, text: str) -> List[float]:
        openai_key = os.environ.get("OPENAI_API_KEY")
        if not openai_key:
            return [0.0] * 1536

        client = OpenAI(api_key=openai_key)
        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=text,
        )
        return response.data[0].embedding

    def get_provider_name(self) -> str:
        return "Friendli AI (meta-llama-3.1-8b-instruct)"

class GeminiService(BaseAIService):
    def __init__(self):
        api_key = os.environ.get("GOOGLE_API_KEY")
        self.has_key = bool(api_key)
        if self.has_key:
            import google.generativeai as genai
            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel('gemini-2.0-flash-exp')
            self.genai = genai

    def fetch_chat_completion(self, prompt: str) -> str:
        if not self.has_key:
            raise ValueError("Gemini API key not configured")

        response = self.model.generate_content(prompt)
        return response.text

    def get_embedding(self, text: str) -> List[float]:
        if self.has_key:
            try:
                result = self.genai.embed_content(
                    model="models/text-embedding-004",
                    content=text
                )
                return result['embedding']
            except:
                pass

        openai_key = os.environ.get("OPENAI_API_KEY")
        if not openai_key:
            return [0.0] * 1536

        client = OpenAI(api_key=openai_key)
        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=text,
        )
        return response.data[0].embedding

    def get_provider_name(self) -> str:
        return "Google Gemini (gemini-2.0-flash-exp)"

class InterviewService:
    def __init__(self, ai_service: BaseAIService):
        self.ai = ai_service

    def generate_content(self, job_position: str):
        prompt = f"""
        Generate realistic interview data for the position: {job_position}.
        Please provide exactly 4 items in the following format:
        [CANDIDATE 1] (A high-quality senior level response)
        [CANDIDATE 2] (A mid-level response with some minor flaws)
        [CANDIDATE 3] (A junior level or slightly weak response)
        [REFERENCE] (The ideal expert answer)

        Ensure each answer is at least 3 sentences long.
        """

        raw_text = self.ai.fetch_chat_completion(prompt)

        c1 = re.search(r"\[CANDIDATE 1\](.*?)(\[CANDIDATE 2\]|\[CANDIDATE 3\]|\[REFERENCE\]|$)", raw_text, re.S)
        c2 = re.search(r"\[CANDIDATE 2\](.*?)(\[CANDIDATE 3\]|\[REFERENCE\]|$)", raw_text, re.S)
        c3 = re.search(r"\[CANDIDATE 3\](.*?)(\[REFERENCE\]|$)", raw_text, re.S)
        ref = re.search(r"\[REFERENCE\](.*?)$", raw_text, re.S)

        transcripts = [
            c1.group(1).strip() if c1 else f"Senior response for {job_position}",
            c2.group(1).strip() if c2 else f"Mid-level response for {job_position}",
            c3.group(1).strip() if c3 else f"Junior response for {job_position}"
        ]
        reference = ref.group(1).strip() if ref else f"Ideal reference for {job_position}"

        return {
            "transcripts": transcripts,
            "reference": reference
        }

class EvaluationService:
    def __init__(self, ai_service: BaseAIService):
        self.ai = ai_service

    def cosine_sim(self, vec_a: List[float], vec_b: List[float]) -> float:
        if not vec_a or not vec_b or len(vec_a) != len(vec_b):
            return 0.0
        dot = sum(a * b for a, b in zip(vec_a, vec_b))
        norm_a = sum(a * a for a in vec_a) ** 0.5
        norm_b = sum(b * b for b in vec_b) ** 0.5
        res = dot / (norm_a * norm_b) if norm_a * norm_b > 0 else 0.0
        return 1.0 if abs(res - 1.0) < 1e-12 else res

    def calculate_rouge(self, candidate: str, reference: str) -> float:
        cand = re.findall(r"\w+", candidate.lower())
        ref = re.findall(r"\w+", reference.lower())
        if not cand or not ref:
            return 0.0
        overlap = len(set(cand) & set(ref))
        return overlap / len(set(ref))

    def calculate_grade(self, score: float) -> str:
        if score >= 0.9:
            return "A"
        if score >= 0.8:
            return "B"
        if score >= 0.7:
            return "C"
        return "D"

    def generate_cross_analysis(self, transcripts: List[str], reference: str, position: str) -> str:
        prompt = f"""
        You are an expert HR analyst. Analyze these {len(transcripts)} candidate interview responses for the {position} position.

        Reference Answer: {reference}

        Candidate Responses:
        {chr(10).join([f"Candidate {i+1}: {t}" for i, t in enumerate(transcripts)])}

        Provide a comprehensive cross-analysis report covering:
        1. Common Strengths: What patterns of excellence appear across candidates?
        2. Common Weaknesses: What gaps or issues are shared among candidates?
        3. Key Differentiators: What makes each candidate unique?
        4. Alignment with Reference: How well do candidates align with the ideal answer?
        5. Hiring Recommendations: Strategic insights for the hiring team.

        Write in a professional, analytical tone suitable for HR decision-makers.
        """

        return self.ai.fetch_chat_completion(prompt)

    def get_hire_decision(self, transcripts: List[str], reference: str) -> HireDecision:
        sims = []
        ref_vec = self.ai.get_embedding(reference)

        for t in transcripts:
            cand_vec = self.ai.get_embedding(t)
            sims.append(self.cosine_sim(cand_vec, ref_vec))

        best_index = sims.index(max(sims))
        candidate_scores = []
        for i, (t, sim) in enumerate(zip(transcripts, sims), start=1):
            rouge = self.calculate_rouge(t, reference)
            overall = (sim + rouge) / 2
            candidate_scores.append(CandidateScore(
                candidate_number=i, cosine_score=sim, rouge_score=rouge,
                overall_score=overall, grade=self.calculate_grade(overall)
            ))

        candidate_list = "\n".join([f"Candidate {i+1}: {t}" for i, t in enumerate(transcripts)])

        prompt = f"""
        As an expert interviewer, compare these {len(transcripts)} candidate responses against the ideal reference answer.
        Reference: {reference}
        {candidate_list}

        Explain why Candidate {best_index + 1} is the best choice based on their technical depth and alignment with the reference.
        """
        explanation = self.ai.fetch_chat_completion(prompt)

        return HireDecision(
            selected_candidate=best_index + 1,
            scores=candidate_scores,
            reason=explanation
        )

    def process_analysis(self, transcripts, reference, position="Unknown"):
        decision = self.get_hire_decision(transcripts, reference)
        cross_analysis = self.generate_cross_analysis(transcripts, reference, position)

        return {
            "report": cross_analysis,
            "score": sum(s.overall_score for s in decision.scores) / len(decision.scores),
            "cosine_score": decision.scores[decision.selected_candidate - 1].cosine_score,
            "rouge_score": decision.scores[decision.selected_candidate - 1].rouge_score,
            "grade": decision.scores[decision.selected_candidate - 1].grade,
            "iterations": [],
            "transcripts": transcripts,
            "reference": reference,
            "hire_decision": decision,
            "ai_provider": self.ai.get_provider_name()
        }