"""
Keeping the user's data on the user's machine.

The app works offline and sends nothing anywhere. Three things back that up:

* :func:`block_network` makes any connection to a non-local address raise, so even a library that tried to phone home
  could not (loopback and local sockets stay allowed: Qt and multiprocessing use them);
* :func:`redact` removes the user's home folder, user name and e-mail addresses from text before it is written to a
  log or a results file, so a results folder can be shared without handing over who made it and where;
* :func:`find_sensitive` is the scanner the build uses (``packaging/audit_package.py``) to refuse an installer that
  contains an e-mail address, an access token, a private key or a path under someone's home folder.
"""
from __future__ import annotations

import getpass
import os
import re
import socket
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional


class NetworkBlocked(OSError):
    """A connection to a non-local address was refused by :func:`block_network`."""


# ── what counts as sensitive ──────────────────────────────────────────────────────────────────────────────────────

EMAIL = re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9\-]+(?:\.[A-Za-z0-9\-]+)*\.[A-Za-z]{2,}")
USER_PATHS = (
    re.compile(r"/home/[^/\s\"'<>:]+/"),
    re.compile(r"/Users/[^/\s\"'<>:]+/"),
    re.compile(r"/mnt/[a-z]/Users/[^/\s\"'<>:]+"),
    re.compile(r"[A-Za-z]:[\\/]+Users[\\/]+[^\\/\s\"'<>:]+", re.I),
)
SECRETS = {
    "github token": re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}"),
    "github fine-grained token": re.compile(r"\bgithub_pat_[A-Za-z0-9_]{20,}"),
    "slack token": re.compile(r"\bxox[abprs]-[A-Za-z0-9\-]{10,}"),
    "aws key": re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    "private key": re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    "api key / password assignment": re.compile(
        r"(?i)\b(?:api[_-]?key|secret|passwd|password|access[_-]?token|auth[_-]?token)\b\s*[:=]\s*['\"][^'\"\s]{8,}['\"]"),
}
# addresses that are placeholders, not people
HARMLESS_EMAILS = ("noreply@", "no-reply@", "@example.com", "@example.org", "@users.noreply.github.com")


@dataclass(frozen=True)
class Finding:
    kind: str
    text: str
    line: int = 0

    def __str__(self) -> str:
        return f"{self.kind}: {self.text!r}" + (f" (line {self.line})" if self.line else "")


def find_sensitive(text: str, extra: Iterable[str] = ()) -> List[Finding]:
    """Every e-mail address, secret, user-home path and any of the ``extra`` strings in ``text``.

    ``extra`` is for identifiers that must not appear at all, such as the author's own address or user name."""
    out: List[Finding] = []
    for no, line in enumerate(text.splitlines() or [text], 1):
        for m in EMAIL.finditer(line):
            if not any(h in m.group(0).lower() for h in HARMLESS_EMAILS):
                out.append(Finding("e-mail address", m.group(0), no))
        for kind, rx in SECRETS.items():
            for m in rx.finditer(line):
                out.append(Finding(kind, m.group(0)[:12] + "...", no))
        merged: List[list] = []
        for a, b in sorted(m.span() for rx in USER_PATHS for m in rx.finditer(line)):
            if merged and a <= merged[-1][1]:                  # /mnt/c/Users/x and /Users/x/ overlap: one finding
                merged[-1][1] = max(merged[-1][1], b)
            else:
                merged.append([a, b])
        for a, b in merged:
            out.append(Finding("path under a user's home folder", line[a:b], no))
        for x in extra:
            if x and x.lower() in line.lower():
                out.append(Finding("forbidden identifier", x, no))
    return out


# ── redaction of what the app writes ──────────────────────────────────────────────────────────────────────────────

def _home_variants() -> List[str]:
    home = str(Path.home())
    variants = {home, home.replace("\\", "/"), home.replace("/", "\\")}
    return sorted((v for v in variants if len(v) > 3), key=len, reverse=True)


def redact(text: str, home: Optional[str] = None, user: Optional[str] = None) -> str:
    """``text`` without the user's home folder, user name in paths, or e-mail addresses.

    ``C:\\Users\\alice\\Videos\\run1.avi`` becomes ``~\\Videos\\run1.avi``; ``/home/alice/x`` becomes ``~/x``. The bare user
    name is only removed inside paths, never from ordinary words."""
    homes = [home] if home else _home_variants()
    for h in homes:
        text = text.replace(h, "~")
    if user is None:
        try:
            user = getpass.getuser()
        except Exception:
            user = ""
    text = re.sub(r"(?i)([A-Za-z]:[\\/]+Users[\\/]+)[^\\/\s\"'<>:]+", r"\1<user>", text)
    text = re.sub(r"(/(?:home|Users)/)[^/\s\"'<>:]+", r"\1<user>", text)
    if user:
        text = re.sub(rf"(?<=[\\/]){re.escape(user)}(?=[\\/])", "<user>", text)
    return EMAIL.sub(lambda m: m.group(0) if any(h in m.group(0).lower() for h in HARMLESS_EMAILS) else "<email>", text)


def short_name(path: str | Path) -> str:
    """Just the file or folder name: what a log should say instead of a full path."""
    return Path(str(path)).name or str(path)


# ── no network ────────────────────────────────────────────────────────────────────────────────────────────────────

_ORIGINAL: dict = {}
_LOOPBACK = ("localhost", "127.0.0.1", "::1", "0.0.0.0", "")


def _is_local(address) -> bool:
    host = address[0] if isinstance(address, (tuple, list)) and address else address
    if isinstance(host, bytes):
        host = host.decode("ascii", "ignore")
    if host is None:
        return True
    host = str(host).strip().lower()
    return host in _LOOPBACK or host.startswith("127.") or host.startswith("::ffff:127.")


def block_network() -> None:
    """From now on this process cannot reach anything but its own machine. Idempotent; undone by
    :func:`unblock_network` (tests)."""
    if _ORIGINAL:
        return
    for name in ("connect", "connect_ex", "sendto"):
        _ORIGINAL[("socket", name)] = getattr(socket.socket, name)
    for name in ("getaddrinfo", "gethostbyname", "gethostbyname_ex", "create_connection"):
        _ORIGINAL[("module", name)] = getattr(socket, name)

    def refuse(what) -> None:
        raise NetworkBlocked(f"network access is switched off in this app (tried to reach {what}); it works offline")

    def guard_method(name):
        real = _ORIGINAL[("socket", name)]

        def method(self, *args, **kw):
            if self.family in (socket.AF_INET, socket.AF_INET6):
                address = args[-1] if name == "sendto" else (args[0] if args else kw.get("address"))
                if not _is_local(address):
                    refuse(address)
            return real(self, *args, **kw)
        method.__name__ = name
        return method

    def guard_host(name):
        real = _ORIGINAL[("module", name)]

        def func(host, *args, **kw):
            if not _is_local(host):
                refuse(host)
            return real(host, *args, **kw)
        func.__name__ = name
        return func

    def guard_connection(address, *args, **kw):
        if not _is_local(address):
            refuse(address)
        return _ORIGINAL[("module", "create_connection")](address, *args, **kw)

    for name in ("connect", "connect_ex", "sendto"):
        setattr(socket.socket, name, guard_method(name))
    for name in ("getaddrinfo", "gethostbyname", "gethostbyname_ex"):
        setattr(socket, name, guard_host(name))
    socket.create_connection = guard_connection
    for var, value in (("HF_HUB_OFFLINE", "1"), ("TRANSFORMERS_OFFLINE", "1"), ("DO_NOT_TRACK", "1"),
                       ("ANONYMIZED_TELEMETRY", "False"), ("TORCH_DISABLE_TELEMETRY", "1")):
        os.environ.setdefault(var, value)


def unblock_network() -> None:
    for (kind, name), real in list(_ORIGINAL.items()):
        setattr(socket.socket if kind == "socket" else socket, name, real)
    _ORIGINAL.clear()


def network_blocked() -> bool:
    return bool(_ORIGINAL)
