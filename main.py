# main.py
import os
import uuid
import math
import datetime
from typing import List, Optional, Dict, Any
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, HTMLResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from supabase import create_client, Client
import requests
from fastapi.middleware.cors import CORSMiddleware




# Load environment variables from .env
load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "openai/gpt-4.1-mini")

if not SUPABASE_URL or not SUPABASE_KEY or not OPENROUTER_API_KEY:
    raise RuntimeError(
        "Missing environment variables. Please set SUPABASE_URL, SUPABASE_KEY, OPENROUTER_API_KEY"
    )

# Initialize Supabase client
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

app = FastAPI(
    title="Chat Backend (single-file) - FastAPI + Supabase + OpenRouter",
    description="Single-file backend implementing chat sessions, knowledge base and LLM pipeline using OpenRouter.",
    version="1.0.0",
    docs_url="/docs",  # we'll override to provide slight customization below
)

# Mount a tiny static directory for custom CSS used by docs
# Create an in-memory path for CSS by writing to disk relative to this file's dir
STATIC_DIR = "static_singlefile"
os.makedirs(STATIC_DIR, exist_ok=True)
CSS_PATH = os.path.join(STATIC_DIR, "swagger_chat.css")
if not os.path.exists(CSS_PATH):
    with open(CSS_PATH, "w", encoding="utf-8") as f:
        f.write(
            """
/* Minimal chat-like tweaks for Swagger UI */
.swagger-ui .topbar { background: #0b5fff; color: white; }
.swagger-ui .info { background: #f5f7ff; border-radius: 8px; padding: 12px; }
body { font-family: Inter, system-ui, -apple-system, "Segoe UI", Roboto, "Helvetica Neue", Arial; }
.swagger-ui .scheme-container { display:none; } /* hide try it auth */
.swagger-ui .opblock-summary { border-radius: 8px; }
"""
        )

app.mount("/static_singlefile", StaticFiles(directory=STATIC_DIR), name="static_singlefile")

origins = [
    "http://localhost:3000",  # your frontend origin (adjust as needed)
    "https://your-wix-site.com",  # Wix domain
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,       # or ["*"] for testing only
    allow_credentials=True,
    allow_methods=["*"],         # allows POST, OPTIONS, GET, etc.
    allow_headers=["*"],         # allows Content-Type, Authorization, etc.
)
class LeadFormRequest(BaseModel):
    session_id: str
    name: str
    phone: str
    email: str

# Pydantic models
class CreateSessionResponse(BaseModel):
    session_id: str


class CreateSessionRequest(BaseModel):
    # Reserved for future expansion; currently no fields required
    meta: Optional[Dict[str, Any]] = None


class ChatRequest(BaseModel):
    session_id: str
    message: str


class ChatResponse(BaseModel):
    reply: str
    session_id: str


class KBAddRequest(BaseModel):
    title: str
    content: str


class KBQueryRequest(BaseModel):
    query: str
    top_k: Optional[int] = Field(3, ge=1, le=20)


class KBEntry(BaseModel):
    id: str
    title: str
    content: str
    score: Optional[float] = None


def is_form_filled(session_id: str) -> bool:
    res = supabase.table("chat_sessions") \
        .select("form_filled") \
        .eq("id", session_id) \
        .single() \
        .execute()

    return res.data.get("form_filled", False)



# -------------------------
# Utility: simple text vectorizer + cosine similarity
# -------------------------
def tokenize(text: str) -> List[str]:
    # Very small tokenizer: lowercase, keep alphanum words
    import re

    tokens = re.findall(r"\b[a-z0-9']+\b", text.lower())
    return tokens


def term_freq(tokens: List[str]) -> Dict[str, float]:
    freqs: Dict[str, float] = {}
    for t in tokens:
        freqs[t] = freqs.get(t, 0.0) + 1.0
    # keep raw frequency (not normalized); cosine will handle scale
    return freqs


def cosine_similarity(tf1: Dict[str, float], tf2: Dict[str, float]) -> float:
    # dot / (||a|| * ||b||)
    dot = 0.0
    for k, v in tf1.items():
        if k in tf2:
            dot += v * tf2[k]
    norm1 = math.sqrt(sum(v * v for v in tf1.values()))
    norm2 = math.sqrt(sum(v * v for v in tf2.values()))
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return dot / (norm1 * norm2)


def rank_documents_by_query(query: str, docs: List[Dict[str, Any]], content_field: str = "content", top_k: int = 3) -> List[KBEntry]:
    q_tokens = tokenize(query)
    q_tf = term_freq(q_tokens)
    scored: List[KBEntry] = []
    for doc in docs:
        text = (doc.get(content_field) or "") + " " + (doc.get("title") or "")
        tf = term_freq(tokenize(text))
        score = cosine_similarity(q_tf, tf)
        scored.append(KBEntry(id=str(doc.get("id")), title=doc.get("title") or "", content=doc.get(content_field) or "", score=score))
    # sort desc by score
    scored.sort(key=lambda x: x.score or 0.0, reverse=True)
    return scored[:top_k]


# -------------------------
# Supabase helpers
# -------------------------
def insert_chat_session(session_id: str) -> Dict[str, Any]:
    now_iso = datetime.datetime.utcnow().isoformat()
    payload = {"id": session_id, "created_at": now_iso}
    res = supabase.table("chat_sessions").insert({"id": session_id}).execute()

    if res.data is None:
        raise Exception("Supabase insert failed")

    return res.data


def insert_chat_message(session_id: str, role: str, message: str):
    response = supabase.table("chat_messages").insert({
        "session_id": session_id,
        "role": role,
        "message": message
    }).execute()

    # SAFE universal parsing
    # 1. If response has .data and it’s not empty → success
    if hasattr(response, "data") and response.data:
        return response.data

    # 2. If response has .model_dump(), then check inside
    if hasattr(response, "model_dump"):
        dumped = response.model_dump()
        if dumped.get("data"):
            return dumped["data"]
        if dumped.get("error"):
            raise HTTPException(500, f"Supabase error: {dumped['error']}")

    # 3. If response has something like .error
    if hasattr(response, "error") and response.error:
        raise HTTPException(500, f"Supabase error: {response.error}")

    # 4. If nothing worked, generic error
    raise HTTPException(500, "Unknown Supabase insert error")



def fetch_chat_history(session_id: str) -> List[Dict[str, Any]]:
    res = (
        supabase.table("chat_messages")
        .select("*")
        .eq("session_id", session_id)
        .order("created_at", desc=False)
        .execute()
    )

    # NEW SDK v2 behavior → everything is inside res.data or res.model_dump()
    if hasattr(res, "data") and res.data is not None:
        return res.data

    # If response follows newer Pydantic model
    if hasattr(res, "model_dump"):
        dump = res.model_dump()
        if "data" in dump and dump["data"] is not None:
            return dump["data"]
        if "error" in dump and dump["error"]:
            raise HTTPException(500, f"Supabase error: {dump['error']}")

    # Older SDK
    if hasattr(res, "error") and res.error:
        raise HTTPException(500, f"Supabase error: {res.error}")

    # Unknown fallback
    raise HTTPException(500, "Unknown Supabase error while fetching history")


def insert_kb_entry(title: str, content: str) -> Dict[str, Any]:
    now_iso = datetime.datetime.utcnow().isoformat()
    payload = {"title": title, "content": content, "created_at": now_iso}

    res = supabase.table("knowledge_base").insert(payload).execute()

    # Check for error
    if hasattr(res, "error") and res.error:
        raise HTTPException(500, f"Supabase error inserting KB: {res.error}")

    if hasattr(res, "model_dump"):
        dump = res.model_dump()
        if dump.get("error"):
            raise HTTPException(500, f"Supabase error inserting KB: {dump['error']}")
        if dump.get("data"):
            return dump["data"]

    return res.data or []


def fetch_all_kb_entries() -> List[Dict[str, Any]]:
    res = supabase.table("knowledge_base").select("*").execute()

    if hasattr(res, "error") and res.error:
        raise HTTPException(500, f"Supabase error fetching KB: {res.error}")

    if hasattr(res, "model_dump"):
        dump = res.model_dump()
        if dump.get("error"):
            raise HTTPException(500, f"Supabase error fetching KB: {dump['error']}")
        return dump.get("data") or []

    return res.data or []


# -------------------------
# OpenRouter LLM call
# -------------------------
def call_openrouter_chat(messages: List[Dict[str, str]], model: str = OPENROUTER_MODEL) -> str:
    """
    Calls OpenRouter chat completions endpoint and returns assistant reply string.
    """
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        # you can add other options here if needed
    }
    resp = requests.post(url, headers=headers, json=payload, timeout=60)
    if resp.status_code >= 400:
        # Try to include response text for debugging
        raise HTTPException(status_code=502, detail=f"OpenRouter error: {resp.status_code} {resp.text}")
    data = resp.json()
    # The OpenRouter response mirrors OpenAI style: data.choices[0].message.content etc.
    # But to be defensive, try multiple shapes.
    try:
        # standard shape
        choices = data.get("choices") or []
        if choices:
            first = choices[0]
            message = first.get("message") or first.get("text") or {}
            if isinstance(message, dict):
                content = message.get("content") or message.get("text") or ""
            else:
                content = str(message)
            return content
    except Exception:
        pass
    # fallback: try top-level content
    if isinstance(data, dict):
        return str(data)
    return ""


# -------------------------
# Routes
# -------------------------
@app.get("/health", response_class=PlainTextResponse)
def health():
    return "ok"

@app.post("/lead_form")
def lead_form(req: LeadFormRequest):
    supabase.table("lead_forms").insert({
        "session_id": req.session_id,
        "name": req.name,
        "phone": req.phone,
        "email": req.email
    }).execute()

    # mark session as form filled
    supabase.table("chat_sessions") \
        .update({"form_filled": True}) \
        .eq("id", req.session_id) \
        .execute()

    return {"status": "form_filled"}

@app.post("/create_session", response_model=CreateSessionResponse)
def create_session():
    session_id = str(uuid.uuid4())
    insert_chat_session(session_id)
    return CreateSessionResponse(session_id=session_id)



@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):

    # Validate session exists
    sess = (
        supabase.table("chat_sessions")
        .select("*")
        .eq("id", req.session_id)
        .execute()
    )

    if not sess.data:  # means no session found
        raise HTTPException(status_code=404, detail="Session not found")

    # Save user message
    insert_chat_message(req.session_id, "user", req.message)

    # Fetch chat history
    history_rows = fetch_chat_history(req.session_id)

    # FORM CHECK
    if not is_form_filled(req.session_id):
        # We let user chat but block the question
        reply = "I'd love to answer that! But before I continue, could you please fill the form? It helps us share accurate details with you."
        insert_chat_message(req.session_id, "assistant", reply)
        return ChatResponse(reply=reply, session_id=req.session_id)

    # Convert to LLM message format
    conversation_messages: List[Dict[str, str]] = []
    for r in history_rows:
        conversation_messages.append({
            "role": r.get("role", "user"),
            "content": r.get("message", "")
        })

    # Fetch KB and rank
    all_kb = fetch_all_kb_entries()
    ranked = rank_documents_by_query(req.message, all_kb, content_field="content", top_k=3)

    kb_texts = []
    for entry in ranked:
        if entry.score and entry.score > 0:
            kb_texts.append(
                f"Title: {entry.title}\nContent: {entry.content}\n---\n"
            )

    # Build system prompt
    system_prompt_lines = [
        "You are a helpful website chatbot specialized for answering user queries about the site.",
        "Always use the knowledge base first if it contains the answer.",
        "If the knowledge base does not contain the answer, use reasoning and your own knowledge.",
        "Be concise and helpful. Include citations to KB entries using the entry title.",
    ]

    if kb_texts:
        system_prompt_lines.append("\nRelevant knowledge base entries (top matches):\n")
        system_prompt_lines.extend(kb_texts)

    system_prompt = "\n".join(system_prompt_lines)

    # Final messages for LLM
    messages_for_llm: List[Dict[str, str]] = [{"role": "system", "content": system_prompt}]
    messages_for_llm.extend(conversation_messages[-50:])
    messages_for_llm.append({"role": "user", "content": req.message})

    # Call OpenRouter
    try:
        assistant_reply = call_openrouter_chat(
            messages_for_llm, model=OPENROUTER_MODEL
        )
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"LLM call failed: {str(e)}")

    # Save assistant message
    insert_chat_message(req.session_id, "assistant", assistant_reply)

    return ChatResponse(reply=assistant_reply, session_id=req.session_id)


@app.get("/history/{session_id}")
def history(session_id: str):
    sess = (
        supabase.table("chat_sessions")
        .select("*")
        .eq("id", session_id)
        .execute()
    )

    if hasattr(sess, "error") and sess.error:
        raise HTTPException(500, f"Supabase error checking session: {sess.error}")

    if hasattr(sess, "model_dump"):
        dump = sess.model_dump()
        if dump.get("error"):
            raise HTTPException(500, f"Supabase error checking session: {dump['error']}")
        if not (dump.get("data") and len(dump["data"]) > 0):
            raise HTTPException(404, "session not found")
    else:
        if not (sess.data and len(sess.data) > 0):
            raise HTTPException(404, "session not found")

    rows = fetch_chat_history(session_id)
    return {"session_id": session_id, "messages": rows}



@app.post("/kb/add")
def kb_add(req: KBAddRequest):
    res = insert_kb_entry(req.title, req.content)
    return {"status": "ok", "inserted": res}


@app.post("/kb/query")
def kb_query(req: KBQueryRequest):
    all_kb = fetch_all_kb_entries()
    ranked = rank_documents_by_query(req.query, all_kb, content_field="content", top_k=req.top_k or 3)
    # Convert to simple dicts
    out = [{"id": e.id, "title": e.title, "content": e.content, "score": e.score} for e in ranked]
    return {"query": req.query, "results": out}


# -------------------------
# Swagger UI customization route override (uses FastAPI's internal helper)
# -------------------------
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.openapi.utils import get_openapi


@app.get("/docs", include_in_schema=False)
def custom_swagger_ui_html():
    openapi_url = app.openapi_url
    swagger_css_url = "/static_singlefile/swagger_chat.css"
    html = get_swagger_ui_html(
        openapi_url=openapi_url,
        title=f"{app.title} - Docs",
        swagger_js_url="https://cdn.jsdelivr.net/npm/swagger-ui-dist@4/swagger-ui-bundle.js",
        swagger_css_url=swagger_css_url,
    )
    # Slight injected HTML to make docs look a bit more like a chat page header
    extra = """
    <style>
        /* Add a small chat-like container above the docs */
        #chat-doc-header { margin: 16px; padding: 12px; border-radius: 8px; background: linear-gradient(90deg,#eef4ff,#fff); box-shadow: 0 1px 4px rgba(12,34,80,0.06);}
        #chat-doc-header h1 { margin: 0; font-size: 18px; }
        #chat-doc-header p { margin: 4px 0 0 0; color: #334155; font-size: 13px; }
    </style>
    <div id="chat-doc-header"><h1>Chat API (Supabase + OpenRouter)</h1><p>Use /create_session → /chat → /history and knowledge base endpoints. Responses are powered by OpenRouter.</p></div>
    """
    full = html.body.decode("utf-8") if hasattr(html.body, "decode") else str(html.body)
    # prepend extra right after opening body tag
    content = html.body
    # Return HTMLResponse with our extra content inserted
    original_html = html.__dict__['_body'].decode("utf-8") if '_body' in html.__dict__ else str(html)
    # Simpler: build new HTML to avoid fragile internals
    openapi_schema = get_openapi(title=app.title, version=app.version, routes=app.routes)
    # Serve a simple swagger UI page referencing our openapi url and css
    page = f"""
    <!DOCTYPE html>
    <html>
    <head>
      <meta charset="UTF-8">
      <title>{app.title} - Docs</title>
      <link rel="stylesheet" type="text/css" href="{swagger_css_url}">
      <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/swagger-ui-dist@4/swagger-ui.css" />
    </head>
    <body>
      <div id="chat-doc-header"><h1>Chat API (Supabase + OpenRouter)</h1><p>Use /create_session → /chat → /history and knowledge base endpoints. Responses are powered by OpenRouter.</p></div>
      <div id="swagger-ui"></div>
      <script src="https://cdn.jsdelivr.net/npm/swagger-ui-dist@4/swagger-ui-bundle.js"></script>
      <script>
      window.onload = function() {{
        const ui = SwaggerUIBundle({{
          url: '{openapi_url}',
          dom_id: '#swagger-ui',
          presets: [SwaggerUIBundle.presets.apis],
          layout: "BaseLayout",
          docExpansion: "none",
          defaultModelsExpandDepth: -1
        }});
        window.ui = ui;
      }};
      </script>
    </body>
    </html>
    """
    return HTMLResponse(content=page)


# Provide the OpenAPI JSON at default path (app.openapi_url) - FastAPI will handle this.
# End of file
