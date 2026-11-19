"""Speech-to-text via Deepgram or OpenAI Whisper API."""
import os
from pathlib import Path

from deepgram import DeepgramClient
from openai import OpenAI

from . import config
from .logging_utils import log_stage

_deepgram_client = None
_openai_client = None


def _get_deepgram_client():
    global _deepgram_client
    if _deepgram_client is None:
        _deepgram_client = DeepgramClient(api_key=config.DEEPGRAM_API_KEY)
    return _deepgram_client


def _get_openai_client():
    global _openai_client
    if _openai_client is None:
        _openai_client = OpenAI(api_key=config.OPENAI_API_KEY)
    return _openai_client


def transcribe_audio(file_path: str) -> str:
    """Transcribe a local audio file to text using the configured STT provider."""
    log_stage("Mic -> STT", input=f"<audio file: {file_path}>")

    if config.STT_PROVIDER == "deepgram":
        return _transcribe_deepgram(file_path)
    else:
        return _transcribe_whisper(file_path)


def _transcribe_deepgram(file_path: str) -> str:
    """Transcribe using Deepgram API (SDK v3)."""
    client = _get_deepgram_client()

    with open(file_path, "rb") as audio_file:
        buffer_data = audio_file.read()

    # Determine encoding from file extension
    ext = Path(file_path).suffix.lower()
    encoding_map = {
        ".webm": "opus",
        ".ogg": "opus",
        ".mp4": "mp4",
        ".m4a": "mp4",
        ".wav": "linear16",
        ".mp3": "mp3",
    }
    encoding = encoding_map.get(ext, "opus")

    response = client.listen.v1.media.transcribe_file(
        request=buffer_data,
        model=config.STT_MODEL,
        encoding=encoding,
        smart_format=True,
        punctuate=True,
        language="en",
    )

    # Extract transcript from Deepgram response (SDK v3 format)
    transcript = ""
    if hasattr(response, 'results') and response.results:
        for channel in response.results.channels:
            if channel.alternatives:
                transcript += channel.alternatives[0].transcript + " "
    elif hasattr(response, 'result') and response.result:
        for channel in response.result.channels:
            if channel.alternatives:
                transcript += channel.alternatives[0].transcript + " "

    text = transcript.strip()
    log_stage("Deepgram -> Orchestrator", output=text)
    return text


def _transcribe_whisper(file_path: str) -> str:
    """Transcribe using OpenAI Whisper API (fallback)."""
    client = _get_openai_client()

    with open(file_path, "rb") as audio_file:
        transcript = client.audio.transcriptions.create(
            model=config.STT_MODEL,
            file=audio_file,
        )

    text = transcript.text.strip()
    log_stage("Whisper -> Orchestrator", output=text)
    return text