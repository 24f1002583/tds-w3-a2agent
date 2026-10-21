import os
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

app = FastAPI()

BASE_URL = os.environ["BASE_URL"]      # e.g. https://myapp.onrender.com/a2a/
MEDIA = "application/a2a+json"

def reply(body, status=200):
    return JSONResponse(body, status_code=status, media_type=MEDIA)

@app.get("/health")
def health():
    return {"ok": True}

@app.get("/.well-known/agent-card.json")
def card():
    return reply({
        "name": "Invoice Action Agent",
        "description": "Reads invoice packages and proposes safe actions.",
        "version": "1.0.0",
        "capabilities": {},
        "supportedInterfaces": [{
            "url": BASE_URL,
            "protocolBinding": "HTTP+JSON",
            "protocolVersion": "1.0",
        }],
        "defaultInputModes": ["application/vnd.ga5.invoice-claim-batch+json"],
        "defaultOutputModes": [
            "application/vnd.ga5.invoice-action-proposals+json",
            "application/vnd.ga5.invoice-action-receipts+json",
        ],
        "skills": [{
            "id": "invoice_action_agent",
            "name": "Invoice action agent",
            "description": "Chooses settle, approve, hold, reject or exception.",
            "tags": ["invoice", "payments"],
        }],
    })

def check_headers(request: Request):
    auth = request.headers.get("authorization", "")
    if not auth.startswith("Bearer ") or not auth[7:].strip():
        return None, reply({"error": "unauthorized"}, 401)
    if request.headers.get("a2a-version") != "1.0":
        return None, reply({"error": "unsupported version"}, 400)
    return auth[7:].strip(), None   # (principal, no error)

@app.get("/a2a/tasks")
def list_tasks(request: Request):
    principal, bad = check_headers(request)
    if bad: return bad
    return reply({"tasks": []})