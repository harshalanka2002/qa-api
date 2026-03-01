import os
import time
import requests
import gradio as gr

BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:9004").rstrip("/")

def ask_api(context, question):
    context = (context or "").strip()
    question = (question or "").strip()

    if not context:
        return "Please paste some context/passage first.", ""
    if not question:
        return "Please type a question.", ""

    t0 = time.time()
    try:
        r = requests.post(
            f"{BACKEND_URL}/qa",
            json={"context": context, "question": question},
            timeout=60,
        )

        # If backend returns non-2xx, show the body
        r.raise_for_status()

        # --- Robust parsing (fix for: 'str' object has no attribute 'get') ---
        data = r.json()
        if isinstance(data, dict):
            answer = data.get("answer", "") or "(No answer.)"
            meta = data.get("meta", "") or ""
        else:
            answer = str(data)
            meta = "Unexpected response format from backend"
        # ---------------------------------------------------------------

        dt = time.time() - t0
        # Add latency to meta (optional but nice)
        if meta:
            meta = f"{meta} | UI roundtrip: {dt:.2f}s"
        else:
            meta = f"UI roundtrip: {dt:.2f}s"

        return answer, meta

    except requests.exceptions.HTTPError:
        # show backend response text for debugging
        return f"Backend HTTP error: {r.status_code} {r.text}", "Request failed"

    except Exception as e:
        return f"Frontend error: {e}", "Request failed"


with gr.Blocks() as demo:
    gr.Markdown("# ❓ Question Answering Assistant (API mode)")
    gr.Markdown(f"Frontend (Gradio) calling backend: `{BACKEND_URL}`")
    gr.Markdown("Paste a passage (context), ask a question, backend calls HuggingFace Inference API.")

    context = gr.Textbox(
        label="Context / Passage",
        lines=10,
        placeholder="Paste paragraph, notes, article excerpt..."
    )
    question = gr.Textbox(
        label="Question",
        lines=2,
        placeholder="Example: What is gravity?"
    )

    btn = gr.Button("Get Answer")

    answer_box = gr.Textbox(label="Answer", lines=4, interactive=False)
    meta_box = gr.Textbox(label="Run info", interactive=False)

    btn.click(ask_api, [context, question], [answer_box, meta_box])

# IMPORTANT: bind to external interface for VM access
demo.launch(server_name="0.0.0.0", server_port=7004)
