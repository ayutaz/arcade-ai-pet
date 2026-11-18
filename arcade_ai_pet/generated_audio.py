"""Blind ASR agreement for generated training sources; no model or device I/O."""
import hashlib
import io
import os
from pathlib import Path
import stat
import unicodedata
import wave
import numpy as np
from .data import SOURCES

PROMPT_LABELS = {text: label for label, texts in SOURCES.items() for text in texts}
READINGS = {
    'あそぼう': 'あそぼ', '遊ぼ': 'あそぼ', '遊ぼう': 'あそぼ',
    '待って': 'まって', 'お休み': 'おやすみ', '有難う': 'ありがとう', 'お出で': 'おいで',
    '歌って': 'うたって', '進む': 'すすむ', '後ろ': 'うしろ', '天気': 'てんき',
    '止まれ': 'とまれ', '頑張れ': 'がんばれ', 'ご飯': 'ごはん', '可愛い': 'かわいい',
    '早く': 'はやく', '速く': 'はやく', '凄い': 'すごい',
    '今日は雨': 'きょうはあめ', '名前は何': 'なまえはなに',
}


def normalize_reading(text):
    if type(text) is not str or len(text) > 512:
        raise ValueError('bounded transcript required')
    text = ''.join(chr(ord(c)-0x60) if '\u30a1' <= c <= '\u30f6' else c
                   for c in unicodedata.normalize('NFKC', text)
                   if not c.isspace() and not unicodedata.category(c).startswith('P'))
    return READINGS.get(text, text)


def _read_wave(path):
    path = Path(path).absolute()
    if path.resolve() != path:
        raise ValueError('real source path required')
    try:
        with os.fdopen(os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK), 'rb') as stream:
            info = os.fstat(stream.fileno())
            if not stat.S_ISREG(info.st_mode) or not 44 <= info.st_size <= 1048576:
                raise ValueError('bounded regular WAV required')
            raw = stream.read(1048577)
        if len(raw) != info.st_size:
            raise ValueError('source changed while reading')
        with wave.open(io.BytesIO(raw), 'rb') as source:
            rate, frames = source.getframerate(), source.getnframes()
            if (source.getnchannels() != 1 or source.getsampwidth() != 2 or
                    source.getcomptype() != 'NONE' or not 8000 <= rate <= 96000 or
                    not rate*.1 <= frames <= rate*2):
                raise ValueError('mono PCM16 within two seconds required')
            data = source.readframes(frames)
        if len(data) != frames*2:
            raise ValueError('incomplete source WAV')
    except (OSError, wave.Error, EOFError):
        raise ValueError('source WAV unavailable or invalid') from None
    samples = np.frombuffer(data, dtype='<i2').astype(np.int16, copy=True)
    if not np.any(samples):
        raise ValueError('silent source cannot prove a spoken label')
    return raw, rate, samples


def audit_generated_wave(path, *, prompt, recognize):
    if type(prompt) is not str or prompt not in PROMPT_LABELS or not callable(recognize):
        raise ValueError('known synthesis prompt and independent recognizer required')
    raw, rate, samples = _read_wave(path)
    magnitude = np.abs(samples.astype(np.int32))
    result = dict(contract='generated-audio-asr-agreement-v1', prompt=prompt, label=PROMPT_LABELS[prompt],
                  wave_sha256=hashlib.sha256(raw).hexdigest(), rate=rate, frames=len(samples),
                  peak=int(magnitude.max()), saturated_frames=int((magnitude >= 32767).sum()),
                  content_check_passed=False, human_transcript_verified=False)
    try:
        # The recognizer sees audio only, never the requested text or label.
        text = recognize(rate, samples.copy())
        normalized = normalize_reading(text)
    except Exception:
        return dict(result, status='recognizer_error', transcript=None, normalized_transcript=None)
    matched = normalized == normalize_reading(prompt)
    return dict(result, status='matched' if matched else 'mismatch', transcript=text,
                normalized_transcript=normalized, content_check_passed=matched)
