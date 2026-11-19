"""Environment configuration for the voice agent.

Gmail and Notion are no longer authenticated via env vars in this file -
both are reached over MCP (see mcp_client.py), and each authenticates
itself using a token cached on disk by a one-time setup command. See
README.md / tutorial.md for that setup.
"""
import os

from dotenv import load_dotenv

load_dotenv()

# LLM Provider (gemini | openai)
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "gemini")

# API Keys
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")

# Model names (with sensible defaults per provider)
if LLM_PROVIDER == "gemini":
    CHAT_MODEL = os.getenv("CHAT_MODEL", "gemini-3.8-flash")
else:
    CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-4o-mini")

STT_PROVIDER = os.getenv("STT_PROVIDER", "deepgram")
STT_MODEL = os.getenv("STT_MODEL", "nova-2")  # Deepgram model

TTS_PROVIDER = os.getenv("TTS_PROVIDER", "elevenlabs")
TTS_MODEL = os.getenv("TTS_MODEL", "eleven_flash_v2_5")  # ElevenLabs model
TTS_VOICE = os.getenv("TTS_VOICE", "21m00Tcm4TlvDq8ikWAM")  # ElevenLabs voice ID (Rachel)

# Guardrails
ENABLE_GUARDRAILS = os.getenv("ENABLE_GUARDRAILS", "true").lower() == "true"
ENABLE_HUMAN_IN_LOOP = os.getenv("ENABLE_HUMAN_IN_LOOP", "true").lower() == "true"


def validate() -> None:
    """Fail fast with a clear message if required secrets are missing."""
    missing = [
        name
        for name, value in [
            ("TAVILY_API_KEY", TAVILY_API_KEY),
        ]
        if not value
    ]

    # LLM provider specific validation
    if LLM_PROVIDER == "gemini":
        if not GOOGLE_API_KEY:
            missing.append("GOOGLE_API_KEY")
    else:
        if not OPENAI_API_KEY:
            missing.append("OPENAI_API_KEY")

    # STT provider specific validation
    if STT_PROVIDER == "deepgram":
        if not DEEPGRAM_API_KEY:
            missing.append("DEEPGRAM_API_KEY")
    else:
        if not OPENAI_API_KEY:
            missing.append("OPENAI_API_KEY (for Whisper)")

    # TTS provider specific validation
    if TTS_PROVIDER == "elevenlabs":
        if not ELEVENLABS_API_KEY:
            missing.append("ELEVENLABS_API_KEY")
    else:
        if not OPENAI_API_KEY:
            missing.append("OPENAI_API_KEY (for TTS)")

    if missing:
        raise RuntimeError(
            f"Missing required environment variables: {', '.join(missing)}. "
            "Copy .env.example to .env and fill them in."
        )