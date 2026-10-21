import os, json, uuid, sqlite3, hashlib, threading
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

import logging, sys
logging.basicConfig(level=logging.INFO, stream=sys.stdout)
log = logging.getLogger("a2a")

app = FastAPI()
BASE_URL = os.environ["BASE_URL"]
MEDIA = "application/a2a+json"
IN_MODE = "application/vnd.ga5.invoice-claim-batch+json"
PROP_MODE = "application/vnd.ga5.invoice-action-proposals+json"
REC_MODE = "application/vnd.ga5.invoice-action-receipts+json"

RES_MODE = "application/vnd.ga5.invoice-action-results+json"

def find_part(msg, media):
    for p in msg.get("parts", []):
        if p.get("mediaType") == media and isinstance(p.get("data"), dict):
            return p["data"]
    return None

def save_task(task):
    db.execute("update tasks set data=? where id=?", (json.dumps(task), task["id"]))

TERMINAL = ("TASK_STATE_COMPLETED", "TASK_STATE_CANCELED")

lock = threading.Lock()
db = sqlite3.connect("state.db", check_same_thread=False)
db.execute("create table if not exists tasks (id text primary key, principal text, data text)")
db.execute("create table if not exists msgs (principal text, mid text, hash text, task_id text, primary key(principal, mid))")
db.commit()

def reply(body, status=200):
    return JSONResponse(body, status_code=status, media_type=MEDIA)

def err(code, msg, status):
    return reply({"error": {"code": code, "message": msg}}, status)

def check_headers(request: Request):
    auth = request.headers.get("authorization", "")
    if not auth.startswith("Bearer ") or not auth[7:].strip():
        return None, err("UNAUTHENTICATED", "unauthorized", 401)
    if request.headers.get("a2a-version") != "1.0":
        return None, err("VERSION_NOT_SUPPORTED", "unsupported version", 400)
    return auth[7:].strip(), None

def canon(x):
    return json.dumps(x, sort_keys=True, separators=(",", ":"))

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
        "supportedInterfaces": [{"url": BASE_URL, "protocolBinding": "HTTP+JSON", "protocolVersion": "1.0"}],
        "defaultInputModes": [IN_MODE],
        "defaultOutputModes": [PROP_MODE, REC_MODE],
        "skills": [{"id": "invoice_action_agent", "name": "Invoice action agent",
                    "description": "Chooses settle, approve, hold, reject or exception.",
                    "tags": ["invoice", "payments"]}],
    })

def load_task(task_id):
    row = db.execute("select principal, data from tasks where id=?", (task_id,)).fetchone()
    return (row[0], json.loads(row[1])) if row else (None, None)

def find_batch(msg):
    for p in msg.get("parts", []):
        d = p.get("data")
        if isinstance(d, dict) and "packages" in d:
            return d
    return None

def continue_task(principal, msg, mid, h):
    owner, task = load_task(msg["taskId"])
    if task is None or owner != principal:
        return err("NOT_FOUND", "not found", 404)
    if msg.get("contextId") != task["contextId"]:
        return err("INVALID_ARGUMENT", "mismatch", 400)
    if task["status"]["state"] in TERMINAL:
        return err("TASK_NOT_ACTIVE", "task already finished", 409)

    stored = task["artifacts"][0]["parts"][0]["data"]
    res = find_part(msg, RES_MODE)
    if res is None or res.get("batchId") != stored["batchId"]:
        return err("INVALID_ARGUMENT", "mismatch", 400)

    by_pkg = {p["packageId"]: p for p in stored["proposals"]}
    seen, executions = set(), []
    for r in res.get("results", []):
        p = by_pkg.get(r.get("packageId"))
        if (p is None or r["packageId"] in seen
                or r.get("actionId") != p["actionId"]
                or r.get("action") != p["action"]
                or r.get("outcome") not in ("ACCEPTED", "REJECTED")):
            return err("INVALID_ARGUMENT", "mismatch", 400)
        seen.add(r["packageId"])
        if r["outcome"] == "ACCEPTED":
            if not r.get("receiptNonce"):
                return err("INVALID_ARGUMENT", "mismatch", 400)
            executions.append({
                "packageId": p["packageId"], "actionId": p["actionId"],
                "action": p["action"], "receiptNonce": r["receiptNonce"],
                "facts": p["facts"], "evidenceRefs": p["evidenceRefs"],
            })

    task["artifacts"].append({
        "artifactId": "art-" + uuid.uuid4().hex,
        "parts": [{"mediaType": REC_MODE,
                   "data": {"batchId": stored["batchId"], "executions": executions}}],
    })
    task["history"].append(msg)
    task["status"] = {"state": "TASK_STATE_COMPLETED"}
    save_task(task)
    db.execute("insert into msgs values (?,?,?,?)", (principal, mid, h, task["id"]))
    db.commit()
    return reply({"task": task})


@app.post("/a2a/message:send")
async def message_send(request: Request):
    principal, bad = check_headers(request)
    if bad:
        return bad
    try:
        body = await request.json()
        msg = body["message"]
        mid = msg["messageId"]
        assert isinstance(msg, dict) and isinstance(mid, str) and mid
    except Exception:
        return err("INVALID_ARGUMENT", "bad request", 400)

    h = hashlib.sha256(canon(msg).encode()).hexdigest()

    with lock:
        row = db.execute("select hash, task_id from msgs where principal=? and mid=?",
                         (principal, mid)).fetchone()
        if row:
            if row[0] != h:
                return err("IDEMPOTENCY_CONFLICT", "message id reused", 409)
            _, task = load_task(row[1])
            return reply({"task": task})

        if msg.get("taskId"):
            return continue_task(principal, msg, mid, h)

        data = find_batch(msg)
        if data is None:
            return err("INVALID_ARGUMENT", "no batch part", 400)

        proposals = []
        for pkg in data.get("packages", []):
            proposals.append({
                "packageId": pkg.get("packageId"),
                "actionId": "act-" + uuid.uuid4().hex,
                "action": "hold_invoice",
                "facts": {"vendorName": "unknown", "invoiceNumber": "unknown",
                          "amountMinor": 0, "currency": "INR"},
                "evidenceRefs": [],
                "rationale": "placeholder " * 10,
            })

        task_id = "task-" + uuid.uuid4().hex
        task = {
            "id": task_id,
            "contextId": "ctx-" + uuid.uuid4().hex,
            "status": {"state": "TASK_STATE_INPUT_REQUIRED"},
            "artifacts": [{
                "artifactId": "art-" + uuid.uuid4().hex,
                "parts": [{"mediaType": PROP_MODE,
                           "data": {"batchId": data.get("batchId"), "proposals": proposals}}],
            }],
            "history": [msg],
        }
        db.execute("insert into tasks values (?,?,?)", (task_id, principal, json.dumps(task)))
        db.execute("insert into msgs values (?,?,?,?)", (principal, mid, h, task_id))
        db.commit()
    return reply({"task": task})

@app.get("/a2a/tasks/{task_id}")
def get_task(task_id: str, request: Request):
    principal, bad = check_headers(request)
    if bad: return bad
    owner, task = load_task(task_id)
    if owner != principal:
        return err("NOT_FOUND", "not found", 404)
    return reply(task)

@app.get("/a2a/tasks")
def list_tasks(request: Request):
    principal, bad = check_headers(request)
    if bad: return bad
    rows = db.execute("select data from tasks where principal=?", (principal,)).fetchall()
    return reply({"tasks": [json.loads(r[0]) for r in rows]})

@app.post("/a2a/tasks/{task_id}:cancel")
async def cancel_task(task_id: str, request: Request):
    principal, bad = check_headers(request)
    if bad:
        return bad
    with lock:
        owner, task = load_task(task_id)
        if task is None or owner != principal:
            return err("NOT_FOUND", "not found", 404)
        if task["status"]["state"] in TERMINAL:
            return err("TASK_NOT_CANCELABLE", "task already finished", 409)
        task["status"] = {"state": "TASK_STATE_CANCELED"}
        save_task(task)
        db.commit()
    return reply(task)