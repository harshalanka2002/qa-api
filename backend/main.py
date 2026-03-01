import os
import time
from typing import Optional, Dict, Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from huggingface_hub import InferenceClient

MODEL_ID = "deepset/bert-base-uncased-squad2"

HF_TOKEN = os.getenv("HF_TOKEN", "").strip()
client = InferenceClient(model=MODEL_ID, token=HF_TOKEN)  # token can be empty for some models

app = FastAPI(title="qa-api backend", version="1.0")


class QARequest(BaseModel):
    context: str
    question: str


class QAResponse(BaseModel):
    answer: str
    meta: Dict[str, Any]


@app.get("/health")
def health():
    return {"ok": True, "mode": "api", "model": MODEL_ID}


@app.post("/qa", response_model=QAResponse)
def qa(req: QARequest):
    context = (req.context or "").strip()
    question = (req.question or "").strip()

    if not context:
        raise HTTPException(status_code=400, detail="Context is required.")
    if not question:
        raise HTTPException(status_code=400, detail="Question is required.")

    t0 = time.time()
    try:
        out = client.question_answering(question=question, context=context)
        dt = time.time() - t0

        answer = out.get("answer", "") or "(No answer found in the provided context.)"
        score = out.get("score", None)
        start = out.get("start", None)
        end = out.get("end", None)

        meta = {
            "mode": "api",
            "model": MODEL_ID,
            "time_sec": round(dt, 4),
        }
        if score is not None:
            meta["score"] = float(score)
        if start is not None and end is not None:
            meta["span"] = {"start": int(start), "end": int(end)}

        return {"answer": answer, "meta": meta}

    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Error calling HF API: {e}")