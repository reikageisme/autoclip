"""Lưu token nhạy cảm (Facebook Page) bằng Windows DPAPI.

Token không nằm trong `.env` hay file JSON đọc được. Trên Windows file được mã hoá
bằng DPAPI theo tài khoản đang chạy; trên hệ điều hành khác (chỉ dùng khi dev/test)
file được ghi quyền 0600 và đánh dấu `PLAIN1` để không giả vờ là đã mã hoá.
"""

from __future__ import annotations

import ctypes
import os
import sys
from ctypes import wintypes
from pathlib import Path

_DPAPI_PREFIX = b"DPAPI1"
_PLAIN_PREFIX = b"PLAIN1"
_ENTROPY = b"autoclip.facebook.v1"


class _DataBlob(ctypes.Structure):
    _fields_ = [
        ("cbData", wintypes.DWORD),
        ("pbData", ctypes.POINTER(ctypes.c_char)),
    ]


def _blob(data: bytes) -> tuple[_DataBlob, ctypes.Array]:
    buffer = ctypes.create_string_buffer(data)
    blob = _DataBlob(len(data), ctypes.cast(buffer, ctypes.POINTER(ctypes.c_char)))
    return blob, buffer


def _win_protect(data: bytes) -> bytes:
    crypt32 = ctypes.WinDLL("crypt32", use_last_error=True)
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    crypt32.CryptProtectData.argtypes = [
        ctypes.POINTER(_DataBlob),
        wintypes.LPCWSTR,
        ctypes.POINTER(_DataBlob),
        ctypes.c_void_p,
        ctypes.c_void_p,
        wintypes.DWORD,
        ctypes.POINTER(_DataBlob),
    ]
    crypt32.CryptProtectData.restype = wintypes.BOOL
    kernel32.LocalFree.argtypes = [wintypes.HLOCAL]
    kernel32.LocalFree.restype = wintypes.HLOCAL

    data_blob, data_buf = _blob(data)
    entropy_blob, entropy_buf = _blob(_ENTROPY)
    out = _DataBlob()
    if not crypt32.CryptProtectData(
        ctypes.byref(data_blob),
        "AutoClip",
        ctypes.byref(entropy_blob),
        None,
        None,
        0,
        ctypes.byref(out),
    ):
        raise OSError(ctypes.get_last_error(), "CryptProtectData thất bại")
    try:
        return ctypes.string_at(out.pbData, out.cbData)
    finally:
        kernel32.LocalFree(out.pbData)
        del data_buf, entropy_buf


def _win_unprotect(data: bytes) -> bytes:
    crypt32 = ctypes.WinDLL("crypt32", use_last_error=True)
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    crypt32.CryptUnprotectData.argtypes = [
        ctypes.POINTER(_DataBlob),
        ctypes.POINTER(wintypes.LPWSTR),
        ctypes.POINTER(_DataBlob),
        ctypes.c_void_p,
        ctypes.c_void_p,
        wintypes.DWORD,
        ctypes.POINTER(_DataBlob),
    ]
    crypt32.CryptUnprotectData.restype = wintypes.BOOL
    kernel32.LocalFree.argtypes = [wintypes.HLOCAL]
    kernel32.LocalFree.restype = wintypes.HLOCAL

    data_blob, data_buf = _blob(data)
    entropy_blob, entropy_buf = _blob(_ENTROPY)
    out = _DataBlob()
    description = wintypes.LPWSTR()
    if not crypt32.CryptUnprotectData(
        ctypes.byref(data_blob),
        ctypes.byref(description),
        ctypes.byref(entropy_blob),
        None,
        None,
        0,
        ctypes.byref(out),
    ):
        raise OSError(ctypes.get_last_error(), "CryptUnprotectData thất bại")
    try:
        return ctypes.string_at(out.pbData, out.cbData)
    finally:
        kernel32.LocalFree(out.pbData)
        del data_buf, entropy_buf


def _protect(data: bytes) -> bytes:
    if sys.platform == "win32":
        return _DPAPI_PREFIX + _win_protect(data)
    return _PLAIN_PREFIX + data


def _unprotect(data: bytes) -> bytes:
    if data.startswith(_DPAPI_PREFIX):
        return _win_unprotect(data[len(_DPAPI_PREFIX) :])
    if data.startswith(_PLAIN_PREFIX):
        return data[len(_PLAIN_PREFIX) :]
    raise ValueError("file secret không đúng định dạng")


class SecretStore:
    """Kho secret nhỏ, mỗi tên một file trong `<data_dir>/secrets/`."""

    def __init__(self, data_dir: Path | str) -> None:
        self.root = Path(data_dir) / "secrets"

    def _path(self, name: str) -> Path:
        safe = "".join(c for c in name if c.isalnum() or c in "._-")
        if not safe or safe != name:
            raise ValueError(f"tên secret không hợp lệ: {name!r}")
        return self.root / f"{safe}.bin"

    def set(self, name: str, value: str) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        target = self._path(name)
        tmp = target.with_suffix(".tmp")
        tmp.write_bytes(_protect(value.encode("utf-8")))
        if sys.platform != "win32":
            os.chmod(tmp, 0o600)
        tmp.replace(target)

    def get(self, name: str) -> str | None:
        path = self._path(name)
        if not path.is_file():
            return None
        try:
            return _unprotect(path.read_bytes()).decode("utf-8")
        except (OSError, ValueError):
            return None

    def has(self, name: str) -> bool:
        return self._path(name).is_file()

    def delete(self, name: str) -> bool:
        path = self._path(name)
        if not path.is_file():
            return False
        path.unlink()
        return True
