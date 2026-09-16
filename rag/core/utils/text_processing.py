"""Decode and normalize extracted document text."""

from __future__ import annotations

import re
import unicodedata
from typing import Final

try:
    from charset_normalizer import from_bytes as _charset_normalizer_from_bytes
except ImportError:  # pragma: no cover - exercised only when the optional dependency is absent
    _charset_normalizer_from_bytes = None

try:
    from ftfy import fix_text as _ftfy_fix_text
except ImportError:  # pragma: no cover - exercised only when the optional dependency is absent
    _ftfy_fix_text = None

_SAMPLE_SIZE_LIMIT: Final[int] = 8192
_MIN_CONFIDENCE_THRESHOLD: Final[float] = 0.8
# Keep normal whitespace and line structure, but remove invisible formatting/control marks.
_INVISIBLE_CHARS = re.compile(
    r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f\u00ad\u061c\u200b-\u200f\u202a-\u202e\u2060-\u2064\u2066-\u206f\ufeff]"
)

# Vietnamese tone marks on the first vowel of oa/oe/uy, commonly found in vbpl.vn text.
_VN_TONE_PAIRS: Final[dict[str, str]] = {
    "òa": "oà", "óa": "oá", "ỏa": "oả", "õa": "oã", "ọa": "oạ",
    "òe": "oè", "óe": "oé", "ỏe": "oẻ", "õe": "oẽ", "ọe": "oẹ",
    "ùy": "uỳ", "úy": "uý", "ủy": "uỷ", "ũy": "uỹ", "ụy": "uỵ",
}
_VN_TONE_MAP: dict[str, str] = {}
for _old, _new in _VN_TONE_PAIRS.items():
    _VN_TONE_MAP[_old] = _new
    _VN_TONE_MAP[_old.upper()] = _new.upper()
    _VN_TONE_MAP[_old.capitalize()] = _new.capitalize()
_VN_TONE_RE = re.compile("|".join(sorted(_VN_TONE_MAP, key=len, reverse=True)))

# BOM-aware UTF encodings come first, followed by encodings commonly found in Vietnamese files.
_FALLBACK_ENCODINGS = ("utf-8-sig", "utf-8", "cp1258", "windows-1258")
_UTF_BOMS = (b"\xef\xbb\xbf", b"\xff\xfe", b"\xfe\xff", b"\xff\xfe\x00\x00", b"\x00\x00\xfe\xff")

_TCVN3_MAP: Final[dict[int, str]] = {
    0x80: "\u00e0", 0x81: "\u1ea3", 0x82: "\u00e3", 0x83: "\u00e1", 0x84: "\u1ea1",  # à, ả, ã, á, ạ
    0x85: "\u0103", 0x86: "\u1eb1", 0x87: "\u1eb3", 0x88: "\u1eb5", 0x89: "\u1eaf",  # ă, ằ, ẳ, ẵ, ắ
    0x8a: "\u1eb7", 0x8b: "\u00e2", 0x8c: "\u1ea7", 0x8d: "\u1ea9", 0x8e: "\u1eab",  # ặ, â, ầ, ẩ, ẫ
    0x8f: "\u1ea5", 0x90: "\u1ead", 0x91: "\u00e8", 0x92: "\u1ebb", 0x93: "\u1ebd",  # ấ, ậ, è, ẻ, ẽ
    0x94: "\u00e9", 0x95: "\u1eb9", 0x96: "\u00ea", 0x97: "\u1ec1", 0x98: "\u1ec3",  # é, ẹ, ê, ề, ể
    0x99: "\u1ec5", 0x9a: "\u1ebf", 0x9b: "\u1ec7", 0x9c: "\u00ec", 0x9d: "\u1ec9",  # ễ, ế, ệ, ì, ỉ
    0x9e: "\u0129", 0x9f: "\u00ed", 0xa0: "\u1ecb", 0xa7: "\u0111",                   # ĩ, í, ị, đ
    0xb5: "\u00f2", 0xb6: "\u1ecf", 0xb7: "\u00f5", 0xb8: "\u00f3", 0xb9: "\u1ecd",  # ò, ỏ, õ, ó, ọ
    0xba: "\u00f4", 0xbb: "\u1ed3", 0xbc: "\u1ed5", 0xbd: "\u1ed7", 0xbe: "\u1ed1",  # ô, ồ, ổ, ỗ, ố
    0xbf: "\u1ed9", 0xc0: "\u01a1", 0xc1: "\u1edd", 0xc2: "\u1edf", 0xc3: "\u1ee1", # ộ, ơ, ờ, ở, ỡ
    0xc4: "\u1edb", 0xc5: "\u1ee3", 0xd9: "\u00f9", 0xda: "\u1ee7", 0xdb: "\u0169",  # ớ, ợ, ù, ủ, ũ
    0xdc: "\u00fa", 0xdd: "\u1ee5", 0xde: "\u01b0", 0xdf: "\u1eeb", 0xe0: "\u1eed",  # ú, ụ, ư, ừ, ử
    0xe1: "\u1eef", 0xe2: "\u1ee9", 0xe3: "\u1ef1", 0xef: "\u1ef3", 0xfd: "\u1ef7",  # ữ, ứ, ự, ỳ, ỷ
    0xfe: "\u1ef9", 0xff: "\u00fd", 0xae: "\u1ef5",                                   # ỹ, ý, ỵ
}

def _decode_tcvn3(data: bytes) -> str:
    """Decode bytes định dạng TCVN3 sang chuỗi Unicode."""
    return "".join(_TCVN3_MAP.get(b, chr(b)) for b in data)


def _is_probable_tcvn3(sample: bytes) -> bool:
    """Heuristic đơn giản: Kiểm tra mật độ các byte đặc trưng chỉ có ở dấu tiếng Việt TCVN3."""
    if not sample:
        return False
    tcvn3_markers = {0x80, 0x83, 0x85, 0x8b, 0x91, 0x94, 0x96, 0xa7, 0xb5, 0xb8, 0xba, 0xc0, 0xde}
    marker_hits = sum(1 for b in sample if b in tcvn3_markers)
    # Nếu tỷ lệ xuất hiện các byte dấu TCVN3 trên mẫu văn bản đủ lớn (>= 1.5%)
    return (marker_hits / len(sample)) >= 0.015


def decode_bytes(data: bytes) -> str:
    """Decode bytes using BOM-aware UTF encodings and Vietnamese fallbacks.

    ``charset-normalizer`` is consulted before the final permissive UTF-8 replacement
    decode. The function never raises for malformed input.
    """
    if data.startswith(_UTF_BOMS[0]):
        data_without_bom = data[len(_UTF_BOMS[0]) :]
        for encoding in ("utf-8", "cp1258"):
            try:
                return data_without_bom.decode(encoding)
            except (UnicodeDecodeError, UnicodeError):
                continue

    if data.startswith((_UTF_BOMS[3], _UTF_BOMS[4])):
        try:
            return data.decode("utf-32")
        except (UnicodeDecodeError, UnicodeError):
            pass

    if data.startswith(_UTF_BOMS[1:3]):
        try:
            return data.decode("utf-16")
        except (UnicodeDecodeError, UnicodeError):
            pass

    for encoding in _FALLBACK_ENCODINGS:
        try:
            return data.decode(encoding)
        except (UnicodeDecodeError, UnicodeError):
            continue
    bytes_sample = data[:_SAMPLE_SIZE_LIMIT]
    if _charset_normalizer_from_bytes is not None:
        try:
            matches = _charset_normalizer_from_bytes(bytes_sample)
            best_match = matches.best()
            if best_match is not None and best_match.encoding and best_match.chaos < (1 - _MIN_CONFIDENCE_THRESHOLD):
                    return data.decode(best_match.encoding)
        except (LookupError, UnicodeError, AttributeError):
            pass
    if _is_probable_tcvn3(bytes_sample):
        try:
            return _decode_tcvn3(data)
        except (UnicodeDecodeError, UnicodeError):
            pass        

    return data.decode("utf-8", errors="replace")


def normalize_text(text: str) -> str:
    """Repair mojibake, normalize Unicode and Vietnamese tone-mark placement."""
    if _ftfy_fix_text is not None:
        text = _ftfy_fix_text(text)

    text = unicodedata.normalize("NFC", text)
    text = _INVISIBLE_CHARS.sub("", text)
    return _VN_TONE_RE.sub(lambda match: _VN_TONE_MAP[match.group()], text)


def decode_and_normalize(value: bytes | bytearray | memoryview | str) -> str:
    """Decode byte input and return cleaned, NFC-normalized text."""
    if isinstance(value, str):
        text = value
    elif isinstance(value, (bytes, bytearray, memoryview)):
        text = decode_bytes(bytes(value))
    else:
        raise TypeError(f"Expected str or bytes-like input, got {type(value).__name__}")
    return normalize_text(text)


# Public alias for callers that prefer the domain term used by the ingestion pipeline.
character_normalizer = decode_and_normalize
