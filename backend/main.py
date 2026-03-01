from fastapi import FastAPI
from pydantic import BaseModel
import os, time
from huggingface_hub import InferenceClient

app = FastAPI(title="qa-api backend")

HF_TOKEN = os.getenv("HF_TOKEN", "").strip()
MODEL_ID = "deepset/bert-base-uncased-squad2"
client = InferenceClient(model=MODEL_ID, token=HF_TOKEN)

class QARequest(BaseModel):
    context: str
    question: str

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/qa")
def qa(req: QARequest):
    context = (req.context or "").strip()
    question = (req.question or "").strip()

    if not context:
        return {"answer": "Please paste some context/passage first.", "meta": "bad_request"}
    if not question:
        return {"answer": "Please type a question.", "meta": "bad_request"}

    t0 = time.time()
    out = client.question_answering(question=question, context=context)
    dt = time.time() - t0

    answer = out.get("answer") or "(No answer found in the provided context.)"
    score = out.get("score", None)
    start = out.get("start", None)
    end = out.get("end", None)

    meta = f"Mode: API | Model: {MODEL_ID} | Time: {dt:.2f}s"
    if score is not None:
        meta += f" | Score: {score:.3f}"
    if start is not None and end is not None:
        meta += f" | Span: {start}-{end}"

    return {"answer": answer, "meta": meta}
