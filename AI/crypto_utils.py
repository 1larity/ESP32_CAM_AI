# crypto_utils.py
# Lightweight symmetric encryption for storing secrets locally.
from __future__ import annotations

import base64
import hashlib
import hmac
import secrets
from pathlib import Path
from typing import Tuple

KEY_PATH = Path(__file__).resolve().parent / "config" / "secret.key"


def _load_or_create_key() -> bytes:
    """
    Load a 32-byte key from disk; create one if missing.
    Stored locally under AI/config/secret.key (gitignored).
    """
    if KEY_PATH.exists():
        data = KEY_PATH.read_bytes()
        return data[:32].ljust(32, b"\0")

    KEY_PATH.parent.mkdir(parents=True, exist_ok=True)
    key = secrets.token_bytes(32)
    KEY_PATH.write_bytes(key)
    return key


def _derive_keystream(key: bytes, nonce: bytes, length: int) -> bytes:
    """
    Derive a pseudo-random keystream using SHA-256 in counter mode.
    Standard library only; sufficient for local secret-at-rest use.
    """
    out = bytearray()
    counter = 0
    while len(out) < length:
        counter_bytes = counter.to_bytes(4, "big")
        out.extend(hashlib.sha256(key + nonce + counter_bytes).digest())
        counter += 1
    return bytes(out[:length])


def _hmac_tag(key: bytes, data: bytes) -> bytes:
    return hmac.new(key, data, hashlib.sha256).digest()


def encrypt(text: str) -> str:
    """
    Encrypt text with a per-machine key.
    Returns base64 string containing nonce + ciphertext + tag.
    """
    key = _load_or_create_key()
    nonce = secrets.token_bytes(12)
    pt = text.encode("utf-8")
    keystream = _derive_keystream(key, nonce, len(pt))
    ct = bytes([a ^ b for a, b in zip(pt, keystream)])
    tag = _hmac_tag(key, nonce + ct)
    return base64.b64encode(nonce + ct + tag).decode("ascii")


def decrypt(token: str) -> Tuple[bool, str]:
    """
    Decrypt token produced by encrypt().
    Returns (ok, plaintext). ok=False if tampered/invalid.
    """
    try:
        raw = base64.b64decode(token.encode("ascii"))
        if len(raw) < 12 + 32:
            return False, ""
        nonce = raw[:12]
        tag = raw[-32:]
        ct = raw[12:-32]

        key = _load_or_create_key()
        if not hmac.compare_digest(_hmac_tag(key, nonce + ct), tag):
            return False, ""

        keystream = _derive_keystream(key, nonce, len(ct))
        pt = bytes([a ^ b for a, b in zip(ct, keystream)])
        return True, pt.decode("utf-8")
    except Exception:
        return False, ""
