import json
import os
from typing import Dict, List

from openai import OpenAI
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

client = OpenAI(api_key="YOUR_OPENAI_API_KEY")
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")


def get_llm_response(prompt: str, model: str = "gpt-4o") -> str:
    try:
        response = client.chat.completions.create(
            model=model, messages=[{"role": "user", "content": prompt}], temperature=0.7
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error: {str(e)}"


def cross_interview_analysis(transcripts: List[str]) -> str:
    combined_text = "\n\n".join(
        [
            f"=== Interview {i+1} ===\n{transcript}"
            for i, transcript in enumerate(transcripts)
        ]
    )
    prompt = f"""
You are an expert consultant analyzing multiple interview transcripts.

Interviews:
{combined_text}

Task:
Provide a comprehensive cross-interview analysis report covering:

1. Common Trends: Key themes mentioned across ALL interviews
2. Diverging Opinions: Points where interviewees disagree or have different perspectives
3. Unique Insights: Noteworthy points mentioned by only one or two candidates
4. Business Recommendations: Actionable insights based on the overall analysis
"""
    return get_llm_response(prompt)


def extract_structured_themes(transcripts: List[str]) -> Dict:
    combined_text = "\n\n".join(
        [f"Interview {i+1}: {transcript}" for i, transcript in enumerate(transcripts)]
    )
    prompt = f"""
Analyze these interviews and extract structured data.

{combined_text}

Return a JSON object with this structure:
{{
    "interviews": [
        {{
            "id": 1,
            "main_topics": ["topic1","topic2"],
            "sentiment": "positive/neutral/negative",
            "key_quote": "representative quote"
        }}
    ],
    "overall_themes": ["theme1","theme2","theme3"]
}}

Return ONLY valid JSON, no extra text.
"""
    response = get_llm_response(prompt)
    try:
        return json.loads(response)
    except:
        return {"interviews": [], "overall_themes": [], "note": "JSON parsing failed"}


def evaluate_cosine_similarity(generated_text: str, reference_text: str) -> float:
    embeddings = embedding_model.encode([generated_text, reference_text])
    similarity = cosine_similarity([embeddings[0]], [embeddings[1]])
    return float(similarity[0][0])


def evaluate_rouge_simple(generated_text: str, reference_text: str) -> Dict[str, float]:
    gen_words = set(generated_text.lower().split())
    ref_words = set(reference_text.lower().split())
    common_words = gen_words.intersection(ref_words)
    precision = len(common_words) / len(gen_words) if gen_words else 0
    recall = len(common_words) / len(ref_words) if ref_words else 0
    f1 = (
        2 * (precision * recall) / (precision + recall)
        if (precision + recall) > 0
        else 0
    )
    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_score": round(f1, 4),
    }


def evaluate_with_multiple_metrics(ai_summary: str, ground_truth: str) -> Dict:
    cosine_score = evaluate_cosine_similarity(ai_summary, ground_truth)
    rouge_scores = evaluate_rouge_simple(ai_summary, ground_truth)
    overall_score = cosine_score * 0.6 + rouge_scores["f1_score"] * 0.4
    return {
        "cosine_similarity": round(cosine_score, 4),
        "rouge_metrics": rouge_scores,
        "overall_quality_score": round(overall_score, 4),
        "performance_grade": get_performance_grade(overall_score),
    }


def get_performance_grade(score: float) -> str:
    if score >= 0.85:
        return "A (Production Ready)"
    elif score >= 0.75:
        return "B (Good Quality)"
    elif score >= 0.65:
        return "C (Needs Improvement)"
    else:
        return "D (Major Revision Required)"


def run_complete_pipeline(interview_files: List[str], ground_truth_path: str):
    print("=" * 70)
    print("ICT GLOBAL INTERNSHIP PORTFOLIO: LLM EVALUATION SYSTEM")
    print("=" * 70)
    print("\n[Step 1] Loading interview transcripts...")
    transcripts = []
    for path in interview_files:
        try:
            with open(path, "r", encoding="utf-8") as f:
                transcripts.append(f.read())
            print(f"  Loaded: {path}")
        except FileNotFoundError:
            print(f"  File not found: {path}")
            return
    print("\n[Step 2] Loading ground truth reference...")
    try:
        with open(ground_truth_path, "r", encoding="utf-8") as f:
            ground_truth = f.read()
        print(f"  Loaded: {ground_truth_path}")
    except FileNotFoundError:
        print(f"  Ground truth file not found: {ground_truth_path}")
        return
    print("\n[Step 3] Performing cross-interview analysis...")
    ai_summary = cross_interview_analysis(transcripts)
    print("  Analysis complete")
    print("\n[Step 4] Extracting structured themes...")
    structured_data = extract_structured_themes(transcripts)
    print(f"  Extracted {len(structured_data.get('overall_themes',[]))} overall themes")
    print("\n[Step 5] Evaluating AI-generated summary...")
    evaluation_results = evaluate_with_multiple_metrics(ai_summary, ground_truth)
    print("\n" + "=" * 70)
    print("ANALYSIS RESULTS")
    print("=" * 70)
    print(ai_summary)
    print("\n" + "=" * 70)
    print("STRUCTURED THEMES (for visualization)")
    print("=" * 70)
    print(json.dumps(structured_data, indent=2, ensure_ascii=False))
    print("\n" + "=" * 70)
    print("PERFORMANCE EVALUATION METRICS")
    print("=" * 70)
    print(f"Cosine Similarity:      {evaluation_results['cosine_similarity']}")
    print(f"ROUGE Precision:        {evaluation_results['rouge_metrics']['precision']}")
    print(f"ROUGE Recall:           {evaluation_results['rouge_metrics']['recall']}")
    print(f"ROUGE F1 Score:         {evaluation_results['rouge_metrics']['f1_score']}")
    print(f"\n>>> Overall Quality Score: {evaluation_results['overall_quality_score']}")
    print(f">>> Performance Grade:     {evaluation_results['performance_grade']}")
    return {
        "summary": ai_summary,
        "structured_data": structured_data,
        "evaluation": evaluation_results,
    }


def run_with_sample_data():
    print("=" * 70)
    print("SAMPLE RUN: LLM Cross-Interview Analysis & Evaluation")
    print("=" * 70)
    sample_transcripts = [
        """
Interviewer: What's your view on remote work?
Candidate A: I strongly believe remote work increases productivity.
When I work from home, I have fewer distractions and can focus better.
However, I think the company needs to invest in better collaboration tools.
""",
        """
Interviewer: What's your view on remote work?
Candidate B: I prefer working in the office because face-to-face interaction
is crucial for team cohesion. Remote work makes communication slower and
can lead to misunderstandings. But I do appreciate the flexibility it offers.
""",
        """
Interviewer: What's your view on remote work?
Candidate C: I think a hybrid model is the best solution.
We should come to the office 2-3 days a week for collaboration,
and work remotely the rest of the time. The key is having proper
infrastructure and clear communication protocols.
""",
    ]
    expert_summary = """
The interviews reveal diverse perspectives on remote work arrangements.
All candidates acknowledge both benefits and challenges of remote work.
Candidate A emphasizes productivity gains, Candidate B prioritizes in-person
collaboration, while Candidate C advocates for a balanced hybrid approach.
A common theme across all interviews is the critical need for improved
collaboration tools and communication infrastructure. The consensus suggests
that a flexible hybrid model with strong technological support would address
most concerns raised by the candidates.
"""
    print("\n[STEP 1] Cross-Interview Analysis\n")
    ai_summary = cross_interview_analysis(sample_transcripts)
    print(ai_summary)
    print("\n[STEP 2] Structured Theme Extraction\n")
    structured = extract_structured_themes(sample_transcripts)
    print(json.dumps(structured, indent=2, ensure_ascii=False))
    print("\n[STEP 3] Performance Evaluation\n")
    evaluation = evaluate_with_multiple_metrics(ai_summary, expert_summary)
    print(f"Cosine Similarity:      {evaluation['cosine_similarity']}")
    print(f"ROUGE F1 Score:         {evaluation['rouge_metrics']['f1_score']}")
    print(f"Overall Quality Score:  {evaluation['overall_quality_score']}")
    print(f"Performance Grade:      {evaluation['performance_grade']}")
    print("\n" + "=" * 70)
    if evaluation["overall_quality_score"] >= 0.75:
        print("RESULT: High quality output. Ready for portfolio demonstration.")
    else:
        print("RESULT: Consider prompt optimization or model fine-tuning.")
    print("=" * 70)


if __name__ == "__main__":
    print("\n Running with sample data \n")
    run_with_sample_data()
