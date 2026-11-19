"""Text-to-speech via ElevenLabs or OpenAI TTS API."""
import tempfile

from elevenlabs import ElevenLabs
from openai import OpenAI

from . import config
from .logging_utils import log_stage

_elevenlabs_client = None
_openai_client = None


def _get_elevenlabs_client():
    global _elevenlabs_client
    if _elevenlabs_client is None:
        _elevenlabs_client = ElevenLabs(api_key=config.ELEVENLABS_API_KEY)
    return _elevenlabs_client


def _get_openai_client():
    global _openai_client
    if _openai_client is None:
        _openai_client = OpenAI(api_key=config.OPENAI_API_KEY)
    return _openai_client


def synthesize_speech(text: str) -> str:
    """Convert text to spoken audio and return the path to an MP3 file."""
    log_stage("Orchestrator -> TTS", input=text)

    if config.TTS_PROVIDER == "elevenlabs":
        return _synthesize_elevenlabs(text)
    else:
        return _synthesize_openai(text)


def _synthesize_elevenlabs(text: str) -> str:
    """Synthesize speech using ElevenLabs API."""
    client = _get_elevenlabs_client()

    tmp_path = tempfile.mktemp(suffix=".mp3")

    # ElevenLabs returns an iterator of chunks
    audio_stream = client.text_to_speech.convert(
        voice_id=config.TTS_VOICE,
        model_id=config.TTS_MODEL,
        text=text,
        output_format="mp3_44100_128",
    )

    with open(tmp_path, "wb") as f:
        for chunk in audio_stream:
            if chunk:
                f.write(chunk)

    log_stage("ElevenLabs -> Speaker", output=f"<audio file: {tmp_path}>")
    return tmp_path


def _synthesize_openai(text: str) -> str:
    """Synthesize speech using OpenAI TTS API (fallback)."""
    client = _get_openai_client()

    tmp_path = tempfile.mktemp(suffix=".mp3")
    with client.audio.speech.with_streaming_response.create(
        model=config.TTS_MODEL,
        voice=config.TTS_VOICE,
        input=text,
    ) as response:
        response.stream_to_file(tmp_path)

    log_stage("OpenAI TTS -> Speaker", output=f"<audio file: {tmp_path}>")
    return tmp_path