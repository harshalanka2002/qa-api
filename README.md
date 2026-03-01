\# qa-api (CS2)



Frontend: Gradio  

Backend: FastAPI  

Mode: API (Hugging Face Inference API)



\## Ports (local / VM)

\- Backend: 9004

\- Frontend: 7004



\## Environment variables

\- HF\_TOKEN (optional depending on model/API access)

\- BACKEND\_URL (frontend) default: http://127.0.0.1:9004



\## Run backend

```bash

pip install -r requirements.txt

uvicorn backend.main:app --host 0.0.0.0 --port 9004

