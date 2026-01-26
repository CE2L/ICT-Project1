import routes
from fastapi import FastAPI, Query, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAIError
from models import InterviewResponse

app = FastAPI(
    title="AI Interview Analyzer API",
    description="LLM-based interview analysis and performance evaluation system",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"status": "AI Interview Analyzer API is running"}

@app.post("/auto-generate", response_model=InterviewResponse)
def auto_generate(
    job_position: str = Query("Frontend Developer", description="Job position for interview generation"),
    provider: str = Query("openai", description="AI engine to use: openai, claude, gemini")
):
    from routes import get_services
    
    int_service, eval_service = get_services(provider)
    
    try:
        generated_data = int_service.generate_content(job_position)
        
        return eval_service.process_analysis(
            transcripts=generated_data["transcripts"],
            reference=generated_data["reference"],
            position=job_position,
        )
    except (OpenAIError, Exception) as e:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Error from {provider} during generation/analysis: {str(e)}",
        )
    except KeyError as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Unexpected AI response format (missing key): {str(e)}",
        )

app.include_router(routes.router)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8004)