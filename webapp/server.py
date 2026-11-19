"""FastAPI web demo: browser mic -> STT -> agent -> Gmail/Notion (MCP) -> TTS -> browser audio.

This wraps the same voice_notion_agent package used by the CLI in main.py.
It exists so the project can be demoed live from a shareable URL (e.g. on
Render) instead of only from a local terminal.

State is kept in a plain in-memory dict keyed by a client-generated session
id. That's fine for a single-instance demo deployment; it is not meant to
survive restarts or scale beyond one process.

SECURITY NOTE: the agent now has real Gmail (read/send) and Notion access
via MCP. If AUTH_PASSWORD is set, every route (including the page itself)
requires HTTP Basic Auth - the browser's native login prompt, checked
against AUTH_USERNAME/AUTH_PASSWORD. Set this before exposing the service
on a public URL, otherwise anyone with the link can read or send from your
real Gmail account.
"""
import base64
import os
import secrets
import tempfile
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from langchain.messages import HumanMessage

from voice_notion_agent import config
from voice_notion_agent.agent import build_agent
from voice_notion_agent.logging_utils import log_stage
from voice_notion_agent.speech_to_text import transcribe_audio
from voice_notion_agent.text_to_speech import synthesize_speech
from voice_notion_agent.guardrails import GuardedAgentWrapper

# Import Google rate limit error for handling
try:
    from langchain_google_genai.chat_models import GoogleRateLimitError
except ImportError:
    GoogleRateLimitError = Exception  # fallback

config.validate()

_agent = None
_sessions: dict[str, list] = {}
_MAX_HISTORY_MESSAGES = 20

_AUTH_USERNAME = os.getenv("AUTH_USERNAME", "demo")
_AUTH_PASSWORD = os.getenv("AUTH_PASSWORD")
_basic_auth = HTTPBasic(auto_error=False)


def _require_auth(credentials: HTTPBasicCredentials | None = Depends(_basic_auth)) -> None:
    """No-op if AUTH_PASSWORD isn't set; otherwise requires matching HTTP Basic Auth.

    auto_error=False on HTTPBasic means a missing Authorization header comes
    back as None instead of an immediate 401 - that's what lets local dev
    (no AUTH_PASSWORD set) skip auth entirely rather than always prompting.
    """
    if not _AUTH_PASSWORD:
        return
    valid = credentials is not None and (
        secrets.compare_digest(credentials.username, _AUTH_USERNAME)
        and secrets.compare_digest(credentials.password, _AUTH_PASSWORD)
    )
    if not valid:
        raise HTTPException(
            status_code=401,
            detail="Invalid credentials.",
            headers={"WWW-Authenticate": "Basic"},
        )


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _agent
    base_agent = await build_agent()

    # Wrap with guardrails if enabled
    if config.ENABLE_GUARDRAILS:
        _agent = GuardedAgentWrapper(base_agent)
    else:
        _agent = base_agent

    yield


app = FastAPI(title="Voice Gmail/Notion Agent Demo", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

STATIC_DIR = Path(__file__).parent / "static"


@app.get("/", dependencies=[Depends(_require_auth)])
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


def _extract_text(content) -> str:
    """Extract plain text from agent response (handles string or list of content blocks)."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        # Handle list of content blocks like [{'type': 'text', 'text': '...'}, ...]
        texts = []
        for block in content:
            if isinstance(block, dict) and block.get('type') == 'text':
                texts.append(block.get('text', ''))
            elif isinstance(block, str):
                texts.append(block)
        return ' '.join(texts)
    return str(content)


async def _run_agent_turn(session_id: str, transcript: str) -> str:
    log_stage("User -> Orchestrator", input=transcript)
    messages = _sessions.setdefault(session_id, [])
    messages.append(HumanMessage(content=transcript))
    try:
        result = await _agent.ainvoke({"messages": messages})
        messages = result["messages"][-_MAX_HISTORY_MESSAGES:]
        _sessions[session_id] = messages
        reply = _extract_text(messages[-1].content)
        log_stage("Orchestrator -> User", output=reply)
        return reply
    except GoogleRateLimitError as exc:
        print(f"[voice] Google rate limit: {exc}")
        return "I'm currently rate-limited (Google free tier exhausted). Please try again later or upgrade your API plan."
    except Exception as exc:
        print(f"[voice] Agent error: {exc}")
        error_str = str(exc)
        if "RESOURCE_EXHAUSTED" in error_str or "429" in error_str:
            return "I'm currently rate-limited. Please try again later or upgrade your API plan."
        elif "quota" in error_str.lower():
            return "API quota exceeded. Please check your billing or try again later."
        else:
            return f"Agent error: {exc}"


@app.post("/api/chat", dependencies=[Depends(_require_auth)])
async def chat(message: str = Form(...), session_id: str = Form(...)) -> JSONResponse:
    """Text-only turn, used by the web UI's fallback text box."""
    reply = await _run_agent_turn(session_id, message)
    return JSONResponse({"transcript": message, "reply": reply})


@app.post("/api/voice", dependencies=[Depends(_require_auth)])
async def voice(audio: UploadFile = File(...), session_id: str = Form(...)) -> JSONResponse:
    """Voice turn: browser sends a recorded clip, we return text + spoken reply."""
    # Keep whatever extension the browser actually sent (webm/ogg/mp4) -
    # Whisper uses it as a hint for which container/codec to decode, so
    # saving everything as .webm regardless of the real format is what
    # causes "could not be decoded" errors on browsers that don't record
    # webm/opus.
    suffix = Path(audio.filename or "command.webm").suffix or ".webm"
    tmp_path = os.path.join(tempfile.gettempdir(), f"{uuid.uuid4()}{suffix}")
    audio_bytes = await audio.read()
    with open(tmp_path, "wb") as f:
        f.write(audio_bytes)

    print(
        f"[voice] received {len(audio_bytes)} bytes, "
        f"filename={audio.filename!r}, content_type={audio.content_type!r}, saved to {tmp_path}"
    )

    try:
        transcript = transcribe_audio(tmp_path)
    except Exception as exc:
        print(f"[voice] transcription failed: {exc}")
        # Return more specific error based on provider
        if config.STT_PROVIDER == "deepgram":
            error_msg = "Speech-to-text failed. Check your Deepgram API key and network."
        else:
            error_msg = "Speech-to-text failed. Check your OpenAI API key and network."
        return JSONResponse(
            {
                "transcript": "",
                "reply": error_msg,
                "audio_base64": None,
            }
        )

    print(f"[voice] transcript: {transcript!r}")

    if not transcript:
        return JSONResponse(
            {"transcript": "", "reply": "I didn't catch that, try again.", "audio_base64": None}
        )

    reply = await _run_agent_turn(session_id, transcript)

    try:
        speech_path = synthesize_speech(reply)
        with open(speech_path, "rb") as f:
            audio_b64 = base64.b64encode(f.read()).decode("ascii")
    except Exception as exc:
        print(f"[voice] TTS failed: {exc}")
        # Provide specific error message based on provider
        if config.TTS_PROVIDER == "elevenlabs":
            error_msg = "Text-to-speech failed. Your ElevenLabs API key is missing the 'text_to_speech' permission. Enable it in your ElevenLabs dashboard (Settings → API Keys)."
        else:
            error_msg = "Text-to-speech failed. Check your OpenAI API key."
        return JSONResponse(
            {"transcript": transcript, "reply": reply, "audio_base64": None, "tts_error": error_msg}
        )

    return JSONResponse({"transcript": transcript, "reply": reply, "audio_base64": audio_b64})
