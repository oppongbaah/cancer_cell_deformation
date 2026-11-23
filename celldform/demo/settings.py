"""
What the app remembers between sessions: which model files were loaded, the last folder, where results go.

Paths only, in one small JSON file in the user's own config folder (``%APPDATA%\\celldform`` on Windows). The files
themselves are never copied, and nothing is sent anywhere. A damaged settings file is ignored, never an error: the app
simply starts as if it were new. Set ``CELLDFORM_HOME`` to keep the settings (and results) in a folder of your choosing,
for example on a USB stick.
"""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Dict, Optional

VERSION = 1


def config_dir() -> Path:
    home = os.environ.get("CELLDFORM_HOME")
    if home:
        return Path(home) / "config"
    try:
        from platformdirs import user_config_dir
        return Path(user_config_dir("celldform", appauthor=False, roaming=True))
    except Exception:
        return Path.home() / ".celldform" / "config"


def default_results_dir() -> Path:
    """Where analysis results go unless the user picks another place: Documents\\celldform\\results."""
    home = os.environ.get("CELLDFORM_HOME")
    if home:
        return Path(home) / "results"
    try:
        from platformdirs import user_documents_dir
        return Path(user_documents_dir()) / "celldform" / "results"
    except Exception:
        return Path.home() / "Documents" / "celldform" / "results"


class Settings:
    """``Settings(path)``; ``path=None`` keeps everything in memory (tests, or a run that must leave no trace)."""

    def __init__(self, path: Optional[Path | str] = None) -> None:
        self.path = Path(path) if path else None
        self.data: dict = {"version": VERSION, "models": {}, "last_dir": "", "results_dir": ""}
        if self.path and self.path.is_file():
            try:
                loaded = json.loads(self.path.read_text(encoding="utf-8"))
                if isinstance(loaded, dict) and loaded.get("version") == VERSION and isinstance(loaded.get("models"), dict):
                    self.data.update({k: loaded[k] for k in ("models", "last_dir", "results_dir") if k in loaded})
            except Exception:
                pass                                         # damaged: start fresh

    @classmethod
    def default(cls) -> "Settings":
        return cls(config_dir() / "settings.json")

    # -- models -------------------------------------------------------------------------------------------------
    def model_paths(self) -> Dict[str, Path]:
        out = {}
        for role, meta in self.data["models"].items():
            if isinstance(meta, dict) and isinstance(meta.get("path"), str) and meta["path"]:
                out[role] = Path(meta["path"])
        return out

    def set_model(self, role: str, path: Path | str, sha256: str = "") -> None:
        self.data["models"][role] = {"path": str(path), "sha256": sha256}
        self.save()

    def forget_models(self) -> None:
        self.data["models"] = {}
        self.save()

    # -- folders ------------------------------------------------------------------------------------------------
    @property
    def last_dir(self) -> str:
        return str(self.data.get("last_dir", ""))

    @last_dir.setter
    def last_dir(self, value: str) -> None:
        self.data["last_dir"] = str(value)
        self.save()

    @property
    def results_dir(self) -> Path:
        return Path(self.data["results_dir"]) if self.data.get("results_dir") else default_results_dir()

    @results_dir.setter
    def results_dir(self, value: Path | str) -> None:
        self.data["results_dir"] = str(value)
        self.save()

    def reset(self) -> None:
        self.data = {"version": VERSION, "models": {}, "last_dir": "", "results_dir": ""}
        self.save()

    # -- saving -------------------------------------------------------------------------------------------------
    def save(self) -> None:
        """Atomic: a crash mid-write cannot leave a half file. Failing to save (read-only disk) is not an error."""
        if not self.path:
            return
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            fd, tmp = tempfile.mkstemp(dir=self.path.parent, suffix=".tmp")
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(self.data, f, indent=2)
            os.replace(tmp, self.path)
        except OSError:
            pass
