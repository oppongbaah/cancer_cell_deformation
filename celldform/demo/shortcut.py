"""
A desktop (or Start Menu) shortcut for the app, for installs done with ``pip`` instead of the Windows installer
(the installer makes its own). It only writes one small file; nothing is downloaded or sent.

    celldform-shortcut                 # desktop shortcut
    celldform-shortcut --start-menu    # Start Menu entry (Windows) / applications menu (Linux)
    celldform-shortcut --remove        # take it away again

Windows: a ``.lnk`` made through PowerShell's WScript.Shell, pointing at ``pythonw.exe -m celldform.demo`` (no console
window) with the app icon. Linux: a ``.desktop`` entry. macOS is not supported.
"""
from __future__ import annotations

import argparse
import os
import stat
import subprocess
import sys
from pathlib import Path
from typing import List, Optional

NAME = "celldform"
COMMENT = "HER2 class of a trapped SKBR3 cell from a recording"


def system_name() -> str:
    return {"win32": "windows", "linux": "linux", "darwin": "macos"}.get(sys.platform, sys.platform)


def icon_path(system: Optional[str] = None) -> Optional[Path]:
    assets = Path(__file__).resolve().parent / "assets"
    for name in (("celldform.ico", "celldform.png") if (system or system_name()) == "windows" else ("celldform.png", "celldform.ico")):
        if (assets / name).is_file():
            return assets / name
    return None


def launch_command(system: Optional[str] = None, python: Optional[str] = None) -> List[str]:
    """What the shortcut runs. A frozen (installer) build is its own program; otherwise this Python, windowless on
    Windows (``pythonw.exe``)."""
    if getattr(sys, "frozen", False):
        return [sys.executable]
    exe = python or sys.executable
    if (system or system_name()) == "windows" and exe.lower().endswith("python.exe"):
        w = exe[:-len("python.exe")] + "pythonw.exe"
        exe = w if (python or Path(w).exists()) else exe
    return [exe, "-m", "celldform.demo"]


def desktop_dir(system: Optional[str] = None) -> Path:
    """The user's desktop (it can be redirected, for example into OneDrive, so Windows is asked)."""
    system = system or system_name()
    if system == "windows":
        try:
            out = subprocess.run(["powershell", "-NoProfile", "-NonInteractive", "-Command", "[Environment]::GetFolderPath('Desktop')"],
                                 capture_output=True, text=True, timeout=20, check=True).stdout.strip()
            if out:
                return Path(out)
        except Exception:
            pass
    elif system == "linux":
        try:
            out = subprocess.run(["xdg-user-dir", "DESKTOP"], capture_output=True, text=True, timeout=10, check=True).stdout.strip()
            if out and Path(out) != Path.home():
                return Path(out)
        except Exception:
            pass
    return Path.home() / "Desktop"


def start_menu_dir(system: Optional[str] = None) -> Path:
    system = system or system_name()
    if system == "windows":
        return Path(os.environ.get("APPDATA", str(Path.home() / "AppData" / "Roaming"))) / "Microsoft" / "Windows" / "Start Menu" / "Programs"
    return Path(os.environ.get("XDG_DATA_HOME", str(Path.home() / ".local" / "share"))) / "applications"


def _ps(text: str) -> str:
    """A PowerShell single-quoted string."""
    return "'" + str(text).replace("'", "''") + "'"


def windows_shortcut_script(lnk: Path, command: List[str], icon: Optional[Path], workdir: Path) -> str:
    """The PowerShell that writes the ``.lnk`` (text only, so it can be checked anywhere)."""
    args = subprocess.list2cmdline(command[1:])
    lines = ["$ErrorActionPreference = 'Stop'", "$s = (New-Object -ComObject WScript.Shell).CreateShortcut(" + _ps(lnk) + ")",
             "$s.TargetPath = " + _ps(command[0]), "$s.Arguments = " + _ps(args), "$s.WorkingDirectory = " + _ps(workdir),
             "$s.Description = " + _ps(COMMENT)]
    if icon:
        lines.append("$s.IconLocation = " + _ps(f"{icon},0"))
    lines.append("$s.Save()")
    return "\n".join(lines)


def linux_desktop_entry(command: List[str], icon: Optional[Path]) -> str:
    def quote(a: str) -> str:
        return '"' + a.replace("\\", "\\\\").replace('"', '\\"').replace("`", "\\`").replace("$", "\\$") + '"' if any(c in a for c in ' \t"\'\\$`') else a
    return "\n".join(["[Desktop Entry]", "Type=Application", f"Name={NAME}", f"Comment={COMMENT}",
                      "Exec=" + " ".join(quote(a) for a in command), *([f"Icon={icon}"] if icon else []),
                      "Terminal=false", "Categories=Science;Education;", "StartupNotify=true", ""])


def shortcut_path(where: Path, system: Optional[str] = None) -> Path:
    return where / (f"{NAME}.lnk" if (system or system_name()) == "windows" else f"{NAME}.desktop")


def create_shortcut(where: Optional[Path] = None, start_menu: bool = False, system: Optional[str] = None,
                    command: Optional[List[str]] = None) -> Path:
    """Write the shortcut and return its path. ``where`` overrides the folder (tests)."""
    system = system or system_name()
    if system not in ("windows", "linux"):
        raise OSError(f"creating a shortcut is not supported on {system}")
    folder = Path(where) if where else (start_menu_dir(system) if start_menu else desktop_dir(system))
    folder.mkdir(parents=True, exist_ok=True)
    target = shortcut_path(folder, system)
    cmd, icon = command or launch_command(system), icon_path(system)
    if system == "linux":
        target.write_text(linux_desktop_entry(cmd, icon), encoding="utf-8")
        target.chmod(target.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP)
        return target
    script = windows_shortcut_script(target, cmd, icon, Path(cmd[0]).parent)
    try:
        subprocess.run(["powershell", "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass", "-Command", script],
                       capture_output=True, text=True, timeout=60, check=True)
    except FileNotFoundError as e:
        raise OSError("PowerShell was not found, so the shortcut could not be made") from e
    except subprocess.CalledProcessError as e:
        raise OSError(f"the shortcut could not be made: {(e.stderr or e.stdout or '').strip()[:200]}") from e
    return target


def remove_shortcut(where: Optional[Path] = None, start_menu: bool = False, system: Optional[str] = None) -> bool:
    system = system or system_name()
    folder = Path(where) if where else (start_menu_dir(system) if start_menu else desktop_dir(system))
    target = shortcut_path(folder, system)
    if target.is_file():
        target.unlink()
        return True
    return False


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="celldform-shortcut", description=__doc__.split("\n\n")[0])
    ap.add_argument("--start-menu", action="store_true", help="Start Menu (Windows) / applications menu (Linux) instead of the desktop")
    ap.add_argument("--remove", action="store_true", help="remove the shortcut")
    ap.add_argument("--dir", default=None, help="put it in this folder instead")
    a = ap.parse_args(argv)
    try:
        if a.remove:
            print("removed" if remove_shortcut(a.dir, a.start_menu) else "no shortcut found")
        else:
            print("created", create_shortcut(a.dir, a.start_menu))
    except OSError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
