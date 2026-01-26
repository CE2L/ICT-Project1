import json
import os
from typing import Dict, List, Optional

import numpy as np
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

try:
    from dotenv import load_dotenv

    load_dotenv()
    print("OK .env file loaded")
except ImportError:
    print("Warning: python-dotenv not installed")

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
GITHUB_API_KEY = os.environ.get("GITHUB_API_KEY")
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
FRIENDLI_API_KEY = os.environ.get("FRIENDLI_API_KEY")
LASTFM_API_KEY = os.environ.get("LASTFM_API_KEY")

SNOWFLAKE_USER = os.environ.get("SNOWFLAKE_USER")
SNOWFLAKE_PASSWORD = os.environ.get("SNOWFLAKE_PASSWORD")
SNOWFLAKE_ACCOUNT = os.environ.get("SNOWFLAKE_ACCOUNT")
SNOWFLAKE_WAREHOUSE = os.environ.get("SNOWFLAKE_WAREHOUSE")
SNOWFLAKE_DATABASE = os.environ.get("SNOWFLAKE_DATABASE")
SNOWFLAKE_SCHEMA = os.environ.get("SNOWFLAKE_SCHEMA")

AWS_ACCESS_KEY = os.environ.get("AWS_ACCESS_KEY")
AWS_SECRET_KEY = os.environ.get("AWS_SECRET_KEY")
AWS_REGION = os.environ.get("AWS_REGION", "ap-northeast-1")
S3_BUCKET_NAME = os.environ.get("S3_BUCKET_NAME")
S3_FILE_KEY = os.environ.get("S3_FILE_KEY")

POSTGRES_HOST = os.environ.get("POSTGRES_HOST")
POSTGRES_PORT = int(os.environ.get("POSTGRES_PORT", 5432))
POSTGRES_DB = os.environ.get("POSTGRES_DB")
POSTGRES_USER = os.environ.get("POSTGRES_USER")
POSTGRES_PASSWORD = os.environ.get("POSTGRES_PASSWORD")

if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable not set")

client = OpenAI(api_key=OPENAI_API_KEY)
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")


def generate_cross_interview_analysis(transcripts: List[str]) -> str:
    formatted_input = ""
    for index, text in enumerate(transcripts):
        formatted_input += f"Interview Data {index+1}:\n{text}\n\n"

    prompt = f"""
Analyze multiple interview transcripts and generate a cross-interview summary report.

Requirements:
1. Extract key trends and metrics common across all interviews
2. Compare opinion differences or conflicting insights between interviewees
3. Present comprehensive synthesis results for overall project direction
4. Suggest specific high-priority improvements

Interview Transcripts:
{formatted_input}

Output Format:
Common Trends
- (list key trends)

Opinion Differences
- (conflicting insights)

Synthesis Insights
- (overall direction)

Priority Improvements
1. (specific suggestions)
"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error: {str(e)}"


def evaluate_performance(ai_output: str, expert_reference: str) -> float:
    embeddings = embedding_model.encode([ai_output, expert_reference])
    score = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
    return float(score)


def evaluate_rouge_metrics(ai_output: str, expert_reference: str) -> Dict[str, float]:
    ai_words = set(ai_output.lower().replace(",", "").replace(".", "").split())
    ref_words = set(expert_reference.lower().replace(",", "").replace(".", "").split())
    common_words = ai_words.intersection(ref_words)
    precision = len(common_words) / len(ai_words) if ai_words else 0
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


def comprehensive_evaluation(ai_output: str, expert_reference: str) -> Dict:
    cosine_score = evaluate_performance(ai_output, expert_reference)
    rouge_metrics = evaluate_rouge_metrics(ai_output, expert_reference)
    overall_score = (cosine_score * 0.6) + (rouge_metrics["f1_score"] * 0.4)

    if overall_score >= 0.85:
        grade = "A (Expert Level)"
        recommendation = "Production ready"
    elif overall_score >= 0.75:
        grade = "B (Good)"
        recommendation = "Minor prompt adjustment recommended"
    elif overall_score >= 0.65:
        grade = "C (Average)"
        recommendation = "Prompt improvement needed"
    else:
        grade = "D (Poor)"
        recommendation = "Pipeline redesign required"

    return {
        "cosine_similarity": round(cosine_score, 4),
        "rouge_precision": rouge_metrics["precision"],
        "rouge_recall": rouge_metrics["recall"],
        "rouge_f1": rouge_metrics["f1_score"],
        "overall_score": round(overall_score, 4),
        "grade": grade,
        "recommendation": recommendation,
    }


def prompt_optimization_experiment(
    transcripts: List[str], expert_reference: str, prompt_versions: List[str]
) -> List[Dict]:
    results = []
    for idx, prompt_template in enumerate(prompt_versions):
        print(
            f"\n[Experiment {idx+1}/{len(prompt_versions)}] Testing prompt version..."
        )
        ai_output = generate_cross_interview_analysis(transcripts)
        evaluation = comprehensive_evaluation(ai_output, expert_reference)
        results.append(
            {
                "version": idx + 1,
                "prompt": prompt_template[:100] + "...",
                "score": evaluation["overall_score"],
                "grade": evaluation["grade"],
                "full_evaluation": evaluation,
            }
        )
    best_result = max(results, key=lambda x: x["score"])
    print(
        f"\nOptimal prompt: Version {best_result['version']} (score: {best_result['score']})"
    )
    return results


def connect_to_snowflake():
    try:
        import snowflake.connector

        conn = snowflake.connector.connect(
            user=os.environ.get("SNOWFLAKE_USER"),
            password=os.environ.get("SNOWFLAKE_PASSWORD"),
            account=os.environ.get("SNOWFLAKE_ACCOUNT"),
            warehouse=os.environ.get("SNOWFLAKE_WAREHOUSE"),
            database=os.environ.get("SNOWFLAKE_DATABASE"),
            schema=os.environ.get("SNOWFLAKE_SCHEMA"),
        )
        print("Snowflake connection successful")
        return conn
    except Exception as e:
        print(f"Snowflake connection failed: {str(e)}")
        return None


def connect_to_postgres():
    try:
        import psycopg2

        conn = psycopg2.connect(
            host=os.environ.get("POSTGRES_HOST"),
            port=int(os.environ.get("POSTGRES_PORT", 5432)),
            database=os.environ.get("POSTGRES_DB"),
            user=os.environ.get("POSTGRES_USER"),
            password=os.environ.get("POSTGRES_PASSWORD"),
        )
        print("PostgreSQL connection successful")
        return conn
    except Exception as e:
        print(f"PostgreSQL connection failed: {str(e)}")
        return None


def save_evaluation_results(
    evaluation_data: Dict, table_name: str = "llm_evaluation_log"
):
    try:
        from datetime import datetime

        import psycopg2

        conn = connect_to_postgres()
        if not conn:
            return False
        cursor = conn.cursor()
        cursor.execute(f"""
CREATE TABLE IF NOT EXISTS {table_name} (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMP,
    cosine_similarity FLOAT,
    rouge_f1 FLOAT,
    overall_score FLOAT,
    grade VARCHAR(50),
    recommendation TEXT
)
""")
        cursor.execute(
            f"""
INSERT INTO {table_name}
(timestamp,cosine_similarity,rouge_f1,overall_score,grade,recommendation)
VALUES (%s,%s,%s,%s,%s,%s)
""",
            (
                datetime.now(),
                evaluation_data["cosine_similarity"],
                evaluation_data["rouge_f1"],
                evaluation_data["overall_score"],
                evaluation_data["grade"],
                evaluation_data["recommendation"],
            ),
        )
        conn.commit()
        cursor.close()
        conn.close()
        print(f"Results saved to: {table_name}")
        return True
    except Exception as e:
        print(f"Save failed: {str(e)}")
        return False


def main_demo():
    print("=" * 70)
    print("ICT Global Internship: Cross-Interview Analysis + Evaluation System")
    print("=" * 70)

    interview_data = [
        "Subject A: Current system has good security but UI is too complex reducing usability. Loading speed improvement is urgent.",
        "Subject B: Security is top priority. But dashboard has too much info and not easy to view at a glance. Performance is satisfactory.",
        "Subject C: UI improvement needed. Too many menus hard to find, page transitions should be faster.",
    ]

    expert_note = """
Interview results show overall satisfaction with security performance but complaints about interface complexity.
Especially among high-end users, page transition speed and navigation simplification are key requirements.
Balancing security and usability is the core challenge for the next project.
"""

    print("\n[STEP 1] Performing cross-interview analysis...\n")
    ai_report = generate_cross_interview_analysis(interview_data)
    print(ai_report)

    print("\n" + "=" * 70)
    print("[STEP 2] AI Generated Report Performance Evaluation")
    print("=" * 70)

    evaluation = comprehensive_evaluation(ai_report, expert_note)

    print("\nEvaluation Results:")
    print(f"  - Cosine Similarity:  {evaluation['cosine_similarity']}")
    print(f"  - ROUGE Precision:    {evaluation['rouge_precision']}")
    print(f"  - ROUGE Recall:       {evaluation['rouge_recall']}")
    print(f"  - ROUGE F1 Score:     {evaluation['rouge_f1']}")
    print(f"\n  Overall Score:        {evaluation['overall_score']}")
    print(f"  Grade:                {evaluation['grade']}")
    print(f"  Recommendation:       {evaluation['recommendation']}")

    print("\n[STEP 3] Attempting to save results...")
    save_evaluation_results(evaluation)

    print("\n" + "=" * 70)
    print("Complete pipeline execution finished")
    print("=" * 70)

    return {"report": ai_report, "evaluation": evaluation}


if __name__ == "__main__":
    main_demo()
