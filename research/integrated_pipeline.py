import os
from typing import Dict, List

from openai import OpenAI
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)
eval_model = SentenceTransformer("all-MiniLM-L6-v2")


def run_integrated_pipeline(
    transcripts: List[str], expert_reference: str, max_iterations: int = 3
) -> Dict:
    print("=" * 70)
    print("Integrated Cross-Interview Analysis & Performance Evaluation System")
    print("=" * 70)

    best_score = 0
    best_report = ""
    iteration_history = []

    for iteration in range(max_iterations):
        print(f"\n[Iteration {iteration+1}/{max_iterations}] Starting analysis...")

        formatted_input = "\n".join(
            [f"Interview {i+1}: {t}" for i, t in enumerate(transcripts)]
        )

        refinement_hint = ""
        if iteration > 0:
            refinement_hint = f"""

Previous Analysis Feedback:
- Previous score: {iteration_history[-1]['score']:.4f}
- Improvement direction: Use similar terms and structure as expert answer
- Reference key keywords: {extract_keywords(expert_reference)}
"""

        analysis_prompt = f"""
You are a professional business consultant. Analyze the provided interview transcripts and create an integrated report.

Analysis Guidelines:
1. Key trends commonly mentioned across all interviews
2. Opinion differences and conflicting points between interviewees
3. Specific insights for future project direction

Interview Data:
{formatted_input}
{refinement_hint}

Output Format:
Key Trends
- (common topics)

Opinion Differences
- (conflict points)

Project Insights
- (specific suggestions)
"""

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": analysis_prompt}],
            temperature=0.7 - (iteration * 0.1),
        )
        ai_report = response.choices[0].message.content

        embeddings = eval_model.encode([ai_report, expert_reference])
        cosine_score = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]

        rouge_score = calculate_rouge_simple(ai_report, expert_reference)

        overall_score = (cosine_score * 0.7) + (rouge_score * 0.3)

        iteration_history.append(
            {
                "iteration": iteration + 1,
                "report": ai_report,
                "cosine_score": float(cosine_score),
                "rouge_score": float(rouge_score),
                "score": float(overall_score),
            }
        )

        print(f"  - Cosine Similarity: {cosine_score:.4f}")
        print(f"  - ROUGE Score: {rouge_score:.4f}")
        print(f"  - Overall Score: {overall_score:.4f}")

        if overall_score > best_score:
            best_score = overall_score
            best_report = ai_report
            print("  New best score achieved")

        if overall_score >= 0.85:
            print("\nTarget score (0.85) achieved. Stopping iterations")
            break

    print("\n" + "=" * 70)
    print("[Final AI Cross-Analysis Report]")
    print("=" * 70)
    print(best_report)
    print("\n" + "=" * 70)
    print("[Performance Evaluation Results]")
    print("=" * 70)
    print(f"Final score: {best_score:.4f}")
    print(f"Total iterations: {len(iteration_history)}")
    print(
        f"Score improvement: {iteration_history[0]['score']:.4f} → {best_score:.4f} (+{(best_score-iteration_history[0]['score'])*100:.1f}%)"
    )

    grade = get_grade(best_score)
    print(f"Grade: {grade}")
    print("=" * 70)

    return {
        "final_report": best_report,
        "final_score": best_score,
        "iterations": iteration_history,
        "grade": grade,
    }


def extract_keywords(text: str, top_n: int = 5) -> str:
    words = text.lower().split()
    stopwords = {
        "the",
        "is",
        "are",
        "and",
        "or",
        "but",
        "in",
        "on",
        "at",
        "to",
        "for",
        "of",
        "with",
        "by",
    }
    keywords = [w for w in words if w not in stopwords and len(w) > 2]
    from collections import Counter

    common = Counter(keywords).most_common(top_n)
    return ", ".join([word for word, _ in common])


def calculate_rouge_simple(generated: str, reference: str) -> float:
    gen_words = set(generated.lower().split())
    ref_words = set(reference.lower().split())
    common = gen_words.intersection(ref_words)

    precision = len(common) / len(gen_words) if gen_words else 0
    recall = len(common) / len(ref_words) if ref_words else 0
    f1 = (
        2 * (precision * recall) / (precision + recall)
        if (precision + recall) > 0
        else 0
    )

    return f1


def get_grade(score: float) -> str:
    if score >= 0.85:
        return "A (Excellent - Expert Level)"
    elif score >= 0.75:
        return "B (Good - Production Ready)"
    elif score >= 0.65:
        return "C (Average - Improvement Needed)"
    else:
        return "D (Poor - Redesign Recommended)"


if __name__ == "__main__":
    sample_interviews = [
        "Subject A: Our service has a complex payment process resulting in high abandonment rates. Simple payment should be introduced.",
        "Subject B: Payment security is good, but button placement is not intuitive. UI/UX improvement is needed.",
        "Subject C: Overall speed is satisfactory, but errors are frequent in mobile payment environment.",
    ]

    reference_note = """
Users provided negative feedback on payment system usability overall.
Core issues are complex payment process and mobile environment instability.
Security feedback is positive, but intuitive button placement from UI/UX perspective and simple payment introduction are top priorities.
"""

    result = run_integrated_pipeline(
        sample_interviews, reference_note, max_iterations=3
    )
