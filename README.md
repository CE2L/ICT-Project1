# AI Cross-Interview Analyzer

LLM-powered system that compares multiple interview answers against an expert reference and produces **quantitative scores + explainable hiring insights**.

[![CI](https://github.com/USERNAME/REPO/workflows/CI/badge.svg)](https://github.com/USERNAME/REPO/actions)
[![Python](https://img.shields.io/badge/Python-3.11%2B-blue)](#)
[![FastAPI](https://img.shields.io/badge/FastAPI-API-green)](#)
[![React](https://img.shields.io/badge/React-18-blue)](#)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](#license)

---

## Table of Contents


- [DEMO LIVE](#DEMO-LIVE)
- [Why this project](#why-this-project)
- [What it does](#what-it-does)
- [Screenshots](#screenshots)
- [Architecture](#architecture)
- [API](#api)
- [Installation](#installation)
  - [Backend Setup](#backend-setup)
  - [Frontend Setup](#frontend-setup)
- [Testing](#testing)
- [Deployment](#deployment)
  - [Using systemd](#using-systemd)
  - [Using Nginx](#using-nginx)
- [Development](#development)
  - [Code Style](#code-style)
  - [Git Workflow](#git-workflow)
- [Tech Stack](#tech-stack)

---
## DEMO LIVE

- **Front End:** [http://4.230.16.126:5174](http://4.230.16.126:5174/)
- **Back End:** [http://4.230.16.126:8004](http://4.230.16.126:8004/)
- **Swagger:** http://4.230.16.126:8004/docs


## Why this project

Interview evaluation is often subjective when comparing multiple candidates.  
This project explores how LLMs can help with **comparison, scoring, and justification** while still keeping outputs measurable.

If you are reviewing a portfolio project, the goal is simple: show that this is not only "LLM text generation," but a workflow that produces **repeatable metrics + decision-ready summaries**.

---

## What it does

### Core workflow
- Accepts **1+ candidate transcripts** plus an **expert reference answer**
- Generates a **cross-candidate analysis report** (trends, differences, recommendations)
- Scores alignment using:
  - **Cosine similarity (embeddings)**  
  - **ROUGE-like overlap score**
- Produces:
  - Overall score + grade
  - Per-candidate scores
  - A **hiring decision** with a short rationale

### Demo workflow (Auto-Generate)
- Generates mock candidates and reference for a given role
- Runs the full analysis immediately
- Useful for:
  - Reviewer demos
  - UI screenshots
  - Quick end-to-end regression checks

---

## Screenshots

### 1) Main Interface (manual input flow)
![Main Interface](screenshots/main-interface.png)

### 2) Auto-Generate Mode (demo-friendly)
![Auto Generate](screenshots/Auto.png)

### 3) Hiring Decision (decision-ready output)
![Hiring Decision](screenshots/Hire.png)

### 4) End-to-End Result Snapshot
![End-to-End](screenshots/END.png)

---

## Tech Stack

### Backend
- **FastAPI** - Modern, high-performance Python web framework
- **OpenAI GPT-4** - LLM for generation and analysis
- **Pydantic** - Data validation and settings management
- **Uvicorn** - ASGI server

### Frontend
- **React 18** - UI library
- **Vite** - Build tool
- **Axios** - HTTP client

### DevOps
- **GitHub Actions** - CI/CD automation
- **Nginx** - Reverse proxy
- **systemd** - Service management
- **pytest** - Testing framework

---

## Architecture
```
├── backend/
│   ├── main.py           # FastAPI application entry point
│   ├── routes.py         # API endpoint definitions
│   ├── services.py       # Business logic layer
│   ├── models.py         # Pydantic data models
│   └── test_main.py      # Unit tests
├── frontend/
│   ├── src/
│   │   ├── App.jsx       # Main React component
│   │   └── App.css       # Styling
├── deployment/
│   ├── systemd.service   # systemd configuration
│   └── nginx.conf        # Nginx reverse proxy config
└── .github/
    └── workflows/
        └── ci.yml        # GitHub Actions CI pipeline
```

---

## API

Swagger UI is available at:
- `http://<HOST>:8004/docs`

### POST `/interviews/generations`
Creates a complete demo dataset (candidates + expert reference) for a job position and runs evaluation immediately.

Use this when you want to test or demonstrate the system without manually writing transcripts.

**Request:**
```json
{
  "job_position": "Frontend Developer",
  "num_candidates": 3
}
```

**Response:**
```json
{
  "report": "Cross-analysis report...",
  "score": 0.47,
  "cosine_score": 0.65,
  "rouge_score": 0.20,
  "grade": "D",
  "transcripts": ["...", "...", "..."],
  "reference": "...",
  "hire_decision": {
    "selected_candidate": 3,
    "reason": "Short explanation...",
    "scores": [
      { "candidate_number": 1, "cosine_score": 0.60, "rouge_score": 0.20, "overall_score": 0.44, "grade": "D" }
    ]
  }
}
```

### POST `/interviews/analyses`
Analyze provided interview transcripts

**Request:**
```json
{
  "transcripts": [
    "Candidate 1 answer...",
    "Candidate 2 answer..."
  ],
  "reference": "Expert reference answer..."
}
```

---

## Installation

### Prerequisites
- Python 3.11+
- Node.js 18+
- OpenAI API key

### Backend Setup
```bash
cd backend
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
export OPENAI_API_KEY='your-api-key-here'
uvicorn main:app --reload --port 8004
```

### Frontend Setup
```bash
cd frontend
npm install
npm run dev
```

---

## Testing
```bash
pytest --cov=. --cov-report=html
```

---

## Deployment

### Using systemd
```bash
sudo cp deployment/systemd.service /etc/systemd/system/interview-analyzer.service
sudo systemctl daemon-reload
sudo systemctl enable interview-analyzer
sudo systemctl start interview-analyzer
```

### Using Nginx
```bash
sudo cp deployment/nginx.conf /etc/nginx/sites-available/interview-analyzer
sudo ln -s /etc/nginx/sites-available/interview-analyzer /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl reload nginx
```

---

## Development

### Code Style
This project uses Black for Python formatting and follows PEP 8 guidelines.

Python code is formatted with Black and isort.  
flake8 is used for linting with team-configurable rules.
```bash
black .
flake8 .
```

### Git Workflow
- Feature branches: `feature/feature-name`
- Bug fixes: `fix/bug-description`
- One task per commit
- Descriptive commit messages

---




## License

MIT License