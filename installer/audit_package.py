"""
Refuse to ship anything that carries the maker's data.

    python installer/audit_package.py PATH [PATH ...] [--own celldform] [--forbid TEXT] [--json]

PATH is a folder (the installer's staging folder, the PyInstaller output), a ``.zip`` or a wheel. The build stops if
this exits non-zero. Two levels of checking, because a big app contains thousands of third-party files whose own
authors' addresses are none of our business:

* **everything shipped** is searched for *this machine's* identifiers: the author's e-mail address and GitHub name, the
  git user's e-mail, and the home folder of the user who built it (``/home/<you>/``, ``C:\\Users\\<you>``). Compiled
  Python files remember the path they were built from, so this also catches a build made on the wrong machine.
* **our own files** (``--own``, default ``celldform``) are also searched for any e-mail address, access token, private
  key or path under any user's home folder, and must not contain data files: model weights, checkpoints, videos,
  tables, results, logs.

The author's name and the project's GitHub name may appear only in the files listed in ``allow_name_in`` (the licence, the
package docstring, the readme and project metadata): that is attribution and the project's address, not a leak.
"""
from __future__ import annotations

import argparse
import getpass
import json
import re
import subprocess
import sys
import tarfile
import zipfile
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Iterable, Iterator, List, Optional, Tuple

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from celldform.demo import privacy  # noqa: E402

TEXT_SUFFIXES = {".py", ".txt", ".md", ".rst", ".json", ".yaml", ".yml", ".toml", ".cfg", ".ini", ".html", ".js", ".css",
                 ".xml", ".iss", ".spec", ".ps1", ".bat", ".cmd", ".sh", ".in", ""}
TEXT_NAMES = ("METADATA", "PKG-INFO", "RECORD", "entry_points.txt", "top_level.txt", "WHEEL", "LICENSE", "NOTICE")
# files that must never be in the app: data, models, secrets, repository internals
FORBIDDEN_SUFFIXES = {".pt", ".pth", ".ckpt", ".joblib", ".pkl", ".pickle", ".h5", ".onnx", ".mp4", ".avi", ".mov", ".mkv",
                      ".tif", ".tiff", ".jpg", ".jpeg", ".xlsx", ".pem", ".key", ".p12", ".pfx", ".log", ".csv"}
FORBIDDEN_NAMES = {".env", "note.txt", "id_rsa", "id_ed25519", "credentials.json", "secrets.json", ".netrc", ".pypirc", "token.txt"}
FORBIDDEN_DIRS = {".git", ".github", "data", "results", "checkpoints", "logs", "supporting_docs", "notebooks", "tests"}
ALLOWED_OWN_DATA = {"celldform.png", "celldform.ico"}           # the icon is data, and ours
MAX_SCAN_BYTES = 40 * 1024 * 1024
# the scanner's own pattern definitions and examples look like what it searches for
PATTERN_FILES = {"celldform/demo/privacy.py", "installer/audit_package.py"}


@dataclass(frozen=True)
class Finding:
    where: str
    what: str

    def __str__(self) -> str:
        return f"{self.where}: {self.what}"


@dataclass
class Report:
    findings: List[Finding] = field(default_factory=list)
    scanned: int = 0

    @property
    def ok(self) -> bool:
        return not self.findings


# ── who must not appear ───────────────────────────────────────────────────────────────────────────────────────────

def machine_identifiers(repo: Path = ROOT) -> Tuple[List[str], List[str]]:
    """``(forbidden, names)``: strings that must not appear anywhere (e-mails, GitHub name, this user's home folder) and the
    author's name(s), allowed only in attribution files."""
    forbidden, names = set(), set()
    try:
        import tomllib
        meta = tomllib.loads((repo / "pyproject.toml").read_text(encoding="utf-8")).get("project", {})
        for a in meta.get("authors", []):
            if a.get("email"):
                forbidden.add(a["email"])
            if a.get("name"):
                names.add(a["name"])
        for url in (meta.get("urls") or {}).values():
            m = re.search(r"github\.com/([^/\s]+)/", str(url))
            if m:                                           # the project's public address: allowed only where it belongs
                names.add(m.group(1))
    except Exception:
        pass
    for key in ("user.email", "user.name"):
        try:
            v = subprocess.run(["git", "-C", str(repo), "config", key], capture_output=True, text=True, timeout=10).stdout.strip()
        except Exception:
            v = ""
        if v:
            (forbidden if key == "user.email" else names).add(v)
    try:
        user = getpass.getuser()
    except Exception:
        user = ""
    if user:
        for pat in (f"/home/{user}/", f"/Users/{user}/", f"/mnt/c/Users/{user}", f"\\Users\\{user}\\", f"/Users/{user}\\"):
            forbidden.add(pat)
    home = str(Path.home())
    if len(home) > 3:
        forbidden.update({home + "/", home + "\\", home.replace("\\", "/") + "/"})
    return sorted(f for f in forbidden if f), sorted(names)


# ── walking what is shipped ───────────────────────────────────────────────────────────────────────────────────────

def _members(path: Path) -> Iterator[Tuple[str, bytes]]:
    """``(relative posix name, content)`` for every file of a folder, zip or wheel (large files are read in part)."""
    if path.is_dir():
        for f in sorted(path.rglob("*")):
            if f.is_file():
                with open(f, "rb") as fh:
                    yield f.relative_to(path).as_posix(), fh.read(MAX_SCAN_BYTES)
    elif zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as z:
            for i in z.infolist():
                if not i.is_dir():
                    with z.open(i) as fh:
                        yield i.filename, fh.read(MAX_SCAN_BYTES)
    elif tarfile.is_tarfile(path):                                   # an sdist (.tar.gz)
        with tarfile.open(path) as t:
            for m in t.getmembers():
                if m.isfile():
                    fh = t.extractfile(m)
                    if fh is not None:
                        yield m.name.split("/", 1)[-1], fh.read(MAX_SCAN_BYTES)   # drop the celldform-0.1.0/ prefix
    else:
        with open(path, "rb") as fh:
            yield path.name, fh.read(MAX_SCAN_BYTES)


def _is_text(name: str, data: bytes) -> bool:
    p = PurePosixPath(name)
    return (p.suffix.lower() in TEXT_SUFFIXES or p.name in TEXT_NAMES) and b"\x00" not in data[:4096]


def _strings(data: bytes, minimum: int = 8) -> str:
    """The printable runs of a binary file, like the ``strings`` command (compiled files hold paths and docstrings)."""
    return "\n".join(m.decode("ascii", "ignore") for m in re.findall(rb"[\x20-\x7e]{%d,}" % minimum, data))


def _own(name: str, own: Iterable[str]) -> bool:
    parts = PurePosixPath(name).parts
    return bool(parts) and (parts[0] in set(own) or (parts[0].endswith(".dist-info") and parts[0].startswith(tuple(f"{o}-" for o in own))))


def audit(paths: Iterable[Path | str], own: Iterable[str] = ("celldform",), forbid: Iterable[str] = (),
          names: Iterable[str] = (), allow_name_in: Iterable[str] = ("LICENSE", "__init__.py", "METADATA", "PKG-INFO", "README.md", "pyproject.toml"),
          identifiers: Optional[Tuple[List[str], List[str]]] = None) -> Report:
    own = tuple(own)
    forbidden, author_names = identifiers if identifiers is not None else machine_identifiers()
    forbidden = [*forbidden, *forbid]
    author_names = [*author_names, *names]
    rep = Report()
    for root in map(Path, paths):
        for name, data in _members(root):
            rep.scanned += 1
            if root.is_dir() and root.name in own:
                name = f"{root.name}/{name}"                       # scanning celldform/ itself: its files are ours
            where = name
            p = PurePosixPath(name)
            mine = _own(name, own)
            if any(part in FORBIDDEN_DIRS for part in p.parts[:-1]) and (mine or p.parts[0] in FORBIDDEN_DIRS):
                rep.findings.append(Finding(where, f"a '{[x for x in p.parts if x in FORBIDDEN_DIRS][0]}' folder must not be shipped"))
            if p.name in FORBIDDEN_NAMES or p.name.startswith(".env"):
                rep.findings.append(Finding(where, "a file that holds secrets or private notes must not be shipped"))
            if mine and p.suffix.lower() in FORBIDDEN_SUFFIXES and p.name not in ALLOWED_OWN_DATA:
                rep.findings.append(Finding(where, f"a {p.suffix} data/model/secret file must not be shipped"))
            text = data.decode("utf-8", "ignore") if _is_text(name, data) else _strings(data)
            for ident in forbidden:
                if ident.lower() in text.lower():
                    rep.findings.append(Finding(where, f"contains {ident!r}, which identifies the maker or the build machine"))
            if mine:                                              # our own files: nothing personal at all
                for f in ([] if name in PATTERN_FILES else privacy.find_sensitive(text)):
                    rep.findings.append(Finding(where, str(f)))
                if p.name not in allow_name_in:
                    for n in author_names:
                        if n and n.lower() in text.lower():
                            rep.findings.append(Finding(where, f"contains the author's name {n!r} (allowed only in {', '.join(allow_name_in)})"))
    seen, unique = set(), []
    for f in rep.findings:
        if (f.where, f.what) not in seen:
            seen.add((f.where, f.what)); unique.append(f)
    rep.findings = unique
    return rep


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--own", action="append", default=None, help="top-level folder(s) that are ours (default: celldform)")
    ap.add_argument("--forbid", action="append", default=[], help="an extra string that must not appear anywhere")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)
    rep = audit(a.paths, own=a.own or ("celldform",), forbid=a.forbid)
    if a.json:
        print(json.dumps({"ok": rep.ok, "scanned": rep.scanned, "findings": [str(f) for f in rep.findings]}, indent=2))
    else:
        for f in rep.findings:
            print("FOUND", f)
        print(f"{'OK' if rep.ok else 'FAILED'}: {rep.scanned} files scanned, {len(rep.findings)} problem(s)")
    return 0 if rep.ok else 1


if __name__ == "__main__":
    sys.exit(main())
