from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import av
from PIL import Image

from cache import CacheStore
from errors import ConversionFailure, UnsupportedMediaError


CONVERSION_VERSION = "media-conversion-v1"


@dataclass(frozen=True)
class PreparedMedia:
    media_id: str
    media_kind: str
    source_path: Path
    prepared_path: Path
    source_sha256: str
    detected_format: str
    mime_type: str
    api_format: str | None
    converted: bool

    def context_metadata(self) -> dict:
        return {
            "media_id": self.media_id,
            "media_kind": self.media_kind,
            "source_file_name": self.source_path.name,
            "detected_format": self.detected_format,
            "detected_mime_type": self.mime_type,
            "converted": self.converted,
            "prepared_format": self.api_format or self.mime_type,
        }


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _convert_avif(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.tmp.png")
    with Image.open(source) as image:
        image.load()
        image.convert("RGB").save(temporary, format="PNG")
    temporary.replace(target)


def _convert_m4a_to_wav(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.tmp.wav")
    input_container = av.open(str(source))
    output_container = av.open(str(temporary), mode="w", format="wav")
    try:
        input_stream = input_container.streams.audio[0]
        output_stream = output_container.add_stream("pcm_s16le", rate=16000)
        output_stream.layout = "mono"
        resampler = av.audio.resampler.AudioResampler(
            format="s16", layout="mono", rate=16000
        )
        for frame in input_container.decode(input_stream):
            for converted_frame in resampler.resample(frame):
                for packet in output_stream.encode(converted_frame):
                    output_container.mux(packet)
        for converted_frame in resampler.resample(None):
            for packet in output_stream.encode(converted_frame):
                output_container.mux(packet)
        for packet in output_stream.encode(None):
            output_container.mux(packet)
    finally:
        input_container.close()
        output_container.close()
    temporary.replace(target)


def prepare_image(media_id: str, path: Path, cache: CacheStore) -> PreparedMedia:
    source_hash = _sha256(path)
    with Image.open(path) as image:
        detected = (image.format or "").upper()
    mime_map = {
        "JPEG": "image/jpeg",
        "PNG": "image/png",
        "WEBP": "image/webp",
        "AVIF": "image/avif",
    }
    if detected not in mime_map:
        raise UnsupportedMediaError("unsupported_media")
    if detected == "AVIF":
        target = cache.converted / f"{media_id}-{source_hash[:12]}.png"
        if not target.exists():
            try:
                _convert_avif(path, target)
            except Exception:
                raise ConversionFailure("conversion_failed") from None
        return PreparedMedia(
            media_id, "image", path, target, source_hash, detected,
            mime_map[detected], None, True
        )
    return PreparedMedia(
        media_id, "image", path, path, source_hash, detected,
        mime_map[detected], None, False
    )


def prepare_audio(media_id: str, path: Path, cache: CacheStore) -> PreparedMedia:
    source_hash = _sha256(path)
    container = av.open(str(path))
    try:
        stream = container.streams.audio[0]
        container_name = container.format.name.casefold()
        codec_name = stream.codec_context.name.casefold()
    finally:
        container.close()

    if "wav" in container_name:
        return PreparedMedia(
            media_id, "voice", path, path, source_hash, "WAV",
            "audio/wav", "wav", False
        )
    if "mp3" in container_name or codec_name.startswith("mp3"):
        return PreparedMedia(
            media_id, "voice", path, path, source_hash, "MP3",
            "audio/mp3", "mp3", False
        )
    if any(name in container_name for name in ("mov", "mp4", "m4a", "3gp", "mj2")):
        target = cache.converted / f"{media_id}-{source_hash[:12]}.wav"
        if not target.exists():
            try:
                _convert_m4a_to_wav(path, target)
            except Exception:
                raise ConversionFailure("conversion_failed") from None
        return PreparedMedia(
            media_id, "voice", path, target, source_hash, "M4A/MP4",
            "audio/mp4", "wav", True
        )
    raise UnsupportedMediaError("unsupported_media")
