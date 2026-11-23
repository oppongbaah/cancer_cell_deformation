"""What ships: the audit that refuses to ship personal data, the desktop shortcut, and a lean install."""
import importlib.util
import json
import stat
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("audit_package", ROOT / "installer" / "audit_package.py")
A = importlib.util.module_from_spec(spec)
sys.modules["audit_package"] = A
spec.loader.exec_module(A)

IDS = (["maker@private-mail.test", "makersgithub", "/home/maker/"], ["Maker Name"])


def tree(tmp, files):
    for name, content in files.items():
        p = tmp / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(content if isinstance(content, bytes) else content.encode())
    return tmp


def run(tmp, files, **kw):
    return A.audit([tree(tmp, files)], identifiers=kw.pop("identifiers", IDS), **kw)


# ── the audit ───────────────────────────────────────────────────────────────────────────────────────────────────

def test_a_clean_tree_passes(tmp_path):
    rep = run(tmp_path, {"celldform/__init__.py": '"""App. Author: Maker Name"""', "celldform/demo/gui.py": "print('hi')",
                         "celldform/demo/assets/celldform.png": b"\x89PNG\x00\x01", "LICENSE": "MIT (c) Maker Name"})
    assert rep.ok and rep.scanned == 4


def test_the_makers_email_is_found_anywhere_and_third_party_emails_are_not(tmp_path):
    rep = run(tmp_path, {"celldform/a.py": "# maker@private-mail.test", "numpy/doc.txt": "numpy-discussion@python.org",
                         "dist-info/METADATA": "Author-email: maker@private-mail.test"})
    kinds = [str(f) for f in rep.findings]
    assert any("celldform/a.py" in k and "maker@private-mail.test" in k for k in kinds)
    assert any("dist-info/METADATA" in k for k in kinds)                           # in any file, even a third party's slot
    assert not any("numpy/doc.txt" in k for k in kinds)                            # somebody else's address is not our leak


def test_the_wheel_metadata_is_checked_as_ours(tmp_path):
    rep = run(tmp_path, {"celldform-0.1.0.dist-info/METADATA": "Author-email: someone@else.io"})
    assert any("e-mail address" in str(f) for f in rep.findings)


def test_the_build_machines_home_folder_is_found_even_inside_compiled_files(tmp_path):
    pyc = b"\x00\x01compiled\x00" + b"/home/maker/work/celldform/demo/gui.py" + b"\x00\x02"
    rep = run(tmp_path, {"celldform/demo/__pycache__/gui.cpython-312.pyc": pyc, "torch/lib/x.dll": b"MZ\x00" + b"/home/maker/build" + b"\x00"})
    assert sum("/home/maker/" in str(f) for f in rep.findings) >= 2


def test_secrets_and_user_paths_in_our_own_files(tmp_path):
    rep = run(tmp_path, {"celldform/a.py": "TOKEN = 'ghp_abcdefghijklmnopqrstuvwxyz0123456789'\npath = '/home/other/x'\nk='AKIAABCDEFGHIJKLMNOP'",
                         "celldform/b.py": "-----BEGIN RSA PRIVATE KEY-----"})
    text = " ".join(str(f) for f in rep.findings)
    assert "github token" in text and "aws key" in text and "private key" in text and "/home/other/" in text


def test_third_party_files_are_not_searched_for_other_peoples_secrets(tmp_path):
    rep = run(tmp_path, {"certifi/test.pem.txt": "-----BEGIN RSA PRIVATE KEY-----", "somelib/x.py": "path='/home/ci/build'"})
    assert rep.ok


def test_data_and_secret_files_must_not_ship(tmp_path):
    rep = run(tmp_path, {"celldform/w.pt": b"x", "celldform/c.joblib": b"x", "celldform/v.mp4": b"x", "celldform/t.csv": "a\n1",
                         ".env": "A=1", "note.txt": "token", "data/frames/a.jpg": b"x", "results/r.txt": "x", "checkpoints/m.txt": "x"})
    text = " ".join(str(f) for f in rep.findings)
    for needle in (".pt data/model", ".joblib", ".mp4", ".csv", "holds secrets", "'data' folder", "'results' folder", "'checkpoints' folder"):
        assert needle in text, needle


def test_a_third_party_package_may_ship_data_files_and_tests_folders(tmp_path):
    rep = run(tmp_path, {"sklearn/datasets/data/iris.csv": "a,b\n1,2", "numpy/tests/test_x.py": "x=1", "PyQt6/Qt6/plugins/x.dll": b"MZ"})
    assert rep.ok


def test_the_authors_name_is_attribution_in_some_files_and_a_leak_in_others(tmp_path):
    ok = run(tmp_path / "a", {"celldform/__init__.py": "Maker Name", "LICENSE": "Maker Name"})
    assert ok.ok
    bad = run(tmp_path / "b", {"celldform/demo/gui.py": "# written by Maker Name"})
    assert any("author's name" in str(f) for f in bad.findings)


def test_a_zip_and_a_wheel_are_audited_too(tmp_path):
    z = tmp_path / "celldform-0.1.0-py3-none-any.whl"
    with zipfile.ZipFile(z, "w") as out:
        out.writestr("celldform/a.py", "ok = 1")
        out.writestr("celldform-0.1.0.dist-info/METADATA", "Author-email: maker@private-mail.test")
    rep = A.audit([z], identifiers=IDS)
    assert not rep.ok and any("METADATA" in f.where for f in rep.findings)


def test_an_sdist_tarball_is_opened_and_audited(tmp_path):
    import io
    import tarfile
    t = tmp_path / "celldform-0.1.0.tar.gz"
    with tarfile.open(t, "w:gz") as out:
        for name, text in (("celldform-0.1.0/celldform/a.py", "ok = 1"), ("celldform-0.1.0/PKG-INFO", "Author-email: maker@private-mail.test")):
            data = text.encode(); info = tarfile.TarInfo(name); info.size = len(data)
            out.addfile(info, io.BytesIO(data))
    rep = A.audit([t], identifiers=IDS)
    assert rep.scanned == 2 and not rep.ok and any("PKG-INFO" in f.where for f in rep.findings)


def test_the_command_line(tmp_path, capsys):
    good = tree(tmp_path / "g", {"celldform/a.py": "x=1"})
    assert A.main([str(good), "--forbid", "nothing-like-this"]) == 0
    assert "OK: 1 files scanned" in capsys.readouterr().out
    bad = tree(tmp_path / "b", {"celldform/a.py": "secret marker-xyz"})
    assert A.main([str(bad), "--forbid", "marker-xyz", "--json"]) == 1
    out = json.loads(capsys.readouterr().out)
    assert out["ok"] is False and out["scanned"] == 1 and "marker-xyz" in out["findings"][0]


def test_this_machines_identifiers_come_from_the_project_and_git():
    forbidden, names = A.machine_identifiers()
    import re
    import tomllib
    meta = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
    assert meta["authors"][0]["name"] in names
    handle = re.search(r"github\.com/([^/]+)/", meta["urls"]["Repository"]).group(1)
    assert handle in names and handle not in forbidden                              # public address: allowed where it belongs
    assert any(f.startswith("/home/") or "Users" in f for f in forbidden)          # the user who builds


def test_the_project_metadata_carries_no_personal_address():
    """What goes into the wheel's METADATA comes from here."""
    import tomllib
    text = (ROOT / "pyproject.toml").read_text()
    meta = tomllib.loads(text)["project"]
    assert all("email" not in a for a in meta["authors"]), "an author e-mail address would be published with the package"
    assert A.privacy.find_sensitive(text) == []
    assert "real-time" not in meta["description"].lower()


def test_the_apps_source_is_clean_once_compiled_files_are_left_out(tmp_path):
    """What the installer is built from: the package without __pycache__ (compiled files remember the build path)."""
    import shutil
    stage = tmp_path / "stage"
    shutil.copytree(ROOT / "celldform", stage / "celldform", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    rep = A.audit([stage])
    assert rep.ok, [str(f) for f in rep.findings][:10]
    assert rep.scanned > 40


def test_compiled_files_in_a_source_tree_would_be_caught(tmp_path):
    import shutil
    stage = tmp_path / "stage"
    shutil.copytree(ROOT / "celldform", stage / "celldform", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    import py_compile
    f = stage / "celldform" / "demo" / "gui.py"
    # a build in the user's home folder bakes the path of the source into the compiled file
    py_compile.compile(str(f), cfile=str(f.with_suffix(".pyc")), dfile=str(Path.home() / "work" / "celldform" / "demo" / "gui.py"), doraise=True)
    rep = A.audit([stage])
    assert not rep.ok and any(".pyc" in x.where for x in rep.findings)


def test_the_icon_is_the_only_data_file_the_package_ships():
    assets = sorted(p.name for p in (ROOT / "celldform" / "demo" / "assets").iterdir())
    assert assets == ["celldform.ico", "celldform.png"]


# ── the desktop shortcut ────────────────────────────────────────────────────────────────────────────────────────

from celldform.demo import shortcut as S  # noqa: E402


def test_linux_entry(tmp_path):
    p = S.create_shortcut(tmp_path / "Desktop", system="linux", command=["/opt/my py/python3", "-m", "celldform.demo"])
    assert p.name == "celldform.desktop" and p.stat().st_mode & stat.S_IXUSR
    text = p.read_text()
    assert "[Desktop Entry]" in text and 'Exec="/opt/my py/python3" -m celldform.demo' in text
    assert "Terminal=false" in text and "Icon=" in text and "celldform.png" in text and "Name=celldform" in text
    assert S.remove_shortcut(tmp_path / "Desktop", system="linux") and not p.exists()
    assert S.remove_shortcut(tmp_path / "Desktop", system="linux") is False          # nothing left: not an error


def test_windows_script_text_is_quoted_safely():
    script = S.windows_shortcut_script(Path("C:/Users/O'Brien/Desktop/celldform.lnk"),
                                       ["C:/Program Files/Py/pythonw.exe", "-m", "celldform.demo"],
                                       Path("C:/x/celldform.ico"), Path("C:/Program Files/Py"))
    assert "CreateShortcut('C:/Users/O''Brien/Desktop/celldform.lnk')" in script                 # apostrophe doubled
    assert "$s.TargetPath = 'C:/Program Files/Py/pythonw.exe'" in script and "$s.Arguments = '-m celldform.demo'" in script
    assert "$s.IconLocation = 'C:/x/celldform.ico,0'" in script and script.rstrip().endswith("$s.Save()")
    assert "$ErrorActionPreference = 'Stop'" in script


def test_windows_creation_calls_powershell_without_a_network_or_a_window(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(S.subprocess, "run", lambda cmd, **k: calls.append((cmd, k)) or subprocess.CompletedProcess(cmd, 0, "", ""))
    p = S.create_shortcut(tmp_path, system="windows", command=["C:/Py/pythonw.exe", "-m", "celldform.demo"])
    assert p == tmp_path / "celldform.lnk"
    cmd, _ = calls[0]
    assert cmd[:4] == ["powershell", "-NoProfile", "-NonInteractive", "-ExecutionPolicy"] and "CreateShortcut" in cmd[-1]


def test_windows_failures_are_explained(tmp_path, monkeypatch):
    def missing(*a, **k):
        raise FileNotFoundError
    monkeypatch.setattr(S.subprocess, "run", missing)
    with pytest.raises(OSError, match="PowerShell was not found"):
        S.create_shortcut(tmp_path, system="windows", command=["x", "-m", "y"])

    def failing(cmd, **k):
        raise subprocess.CalledProcessError(1, cmd, stderr="access denied")
    monkeypatch.setattr(S.subprocess, "run", failing)
    with pytest.raises(OSError, match="could not be made: access denied"):
        S.create_shortcut(tmp_path, system="windows", command=["x", "-m", "y"])


def test_other_systems_are_refused_clearly(tmp_path):
    with pytest.raises(OSError, match="not supported on macos"):
        S.create_shortcut(tmp_path, system="macos")


def test_the_launch_command_is_windowless_on_windows_and_frozen_builds_launch_themselves(monkeypatch):
    assert S.launch_command("windows", python="C:/Py/python.exe") == ["C:/Py/pythonw.exe", "-m", "celldform.demo"]
    assert S.launch_command("linux", python="/usr/bin/python3") == ["/usr/bin/python3", "-m", "celldform.demo"]
    monkeypatch.setattr(S.sys, "frozen", True, raising=False)
    assert S.launch_command("windows") == [S.sys.executable]


def test_folders(monkeypatch, tmp_path):
    monkeypatch.setenv("APPDATA", str(tmp_path / "Roaming"))
    assert S.start_menu_dir("windows") == tmp_path / "Roaming" / "Microsoft" / "Windows" / "Start Menu" / "Programs"
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "share"))
    assert S.start_menu_dir("linux") == tmp_path / "share" / "applications"
    assert S.icon_path("windows").name == "celldform.ico" and S.icon_path("linux").name == "celldform.png"


def test_the_shortcut_command(tmp_path, capsys):
    assert S.main(["--dir", str(tmp_path / "d")]) in (0, 1)
    if sys.platform == "linux":
        assert (tmp_path / "d" / "celldform.desktop").is_file()
        assert S.main(["--dir", str(tmp_path / "d"), "--remove"]) == 0
        assert not (tmp_path / "d" / "celldform.desktop").exists()
        assert S.main(["--dir", str(tmp_path / "d"), "--remove"]) == 0 and "no shortcut found" in capsys.readouterr().out


# ── a lean install ──────────────────────────────────────────────────────────────────────────────────────────────

def test_the_app_starts_without_the_research_only_libraries(tmp_path):
    """The Windows build leaves out nnU-Net, napari, statsmodels and the rest: the app must not need them."""
    code = r'''
import sys, importlib.abc
BLOCK = {"napari", "nnunetv2", "statsmodels", "torchvision", "fastapi", "uvicorn", "mpi4py", "mkdocs", "SimpleITK",
         "batchgenerators", "dynamic_network_architectures", "requests", "urllib3", "yaml_dummy"}
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path, target=None):
        if name.split(".")[0] in BLOCK:
            raise ImportError("blocked for the lean-install test: " + name)
sys.meta_path.insert(0, Block())
import os; os.environ["QT_QPA_PLATFORM"] = "offscreen"
from celldform.demo import gui, run, dashboard, player, segment, framework, models, privacy, settings, shortcut, session
from celldform.demo.qt import QtWidgets
app = QtWidgets.QApplication([])
w = gui.MainWindow()
print("OK", sorted(m for m in sys.modules if m.split(".")[0] in BLOCK))
'''
    import os
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(p for p in (str(ROOT), os.environ.get("PYTHONPATH", "")) if p)}
    r = subprocess.run([sys.executable, "-s", "-c", code], capture_output=True, text=True, timeout=180, cwd=ROOT, env=env)
    assert r.returncode == 0 and r.stdout.strip().startswith("OK []"), r.stderr[-1500:]


# ── the Windows recipe (cannot be built here: these check that it contains the safeguards) ────────────────────────

WIN = ROOT / "installer" / "windows"


def read(name):
    return (WIN / name).read_text(encoding="utf-8")


def test_the_installer_folder_carries_nothing_personal(tmp_path):
    import shutil
    shutil.copytree(ROOT / "installer", tmp_path / "installer", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    rep = A.audit([tmp_path / "installer"], own=("installer",), identifiers=A.machine_identifiers())
    assert rep.ok, [str(f) for f in rep.findings][:8]


def test_the_inno_script_installs_for_the_user_with_a_desktop_shortcut_and_a_neutral_publisher():
    iss = read("celldform.iss")
    assert "PrivilegesRequired=lowest" in iss                                  # no administrator needed
    assert 'Name: "desktopicon"' in iss and "Tasks: desktopicon" in iss and "{autodesktop}" in iss
    assert 'Name: "startmenuicon"' in iss and "{autoprograms}" in iss
    assert "AppPublisher={#AppName}" in iss                                    # nothing about the maker
    assert "AppPublisherURL" not in iss and "AppContact" not in iss and "AppSupportURL" not in iss
    import re
    assert re.search(r"AppId=\{\{[0-9A-F]{8}-([0-9A-F]{4}-){3}[0-9A-F]{12}\}", iss)
    assert "[UninstallDelete]" not in iss and "Documents" in iss and "never deleted" in iss    # results are never removed
    assert "OutputBaseFilename=celldform-setup-{#AppVersion}" in iss and "LicenseFile=..\\..\\LICENSE" in iss


def test_the_inno_default_version_matches_the_project():
    import re
    import tomllib
    v = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]["version"]
    assert re.search(r'#define AppVersion "' + re.escape(v) + '"', read("celldform.iss"))


def test_the_pyinstaller_recipe_is_windowed_unpacked_and_lean():
    spec = read("celldform.spec")
    assert "console=False" in spec and "upx=False" in spec and "uac_admin=False" in spec
    for lib in ("napari", "statsmodels", "torchvision", "requests", "urllib3", "SimpleITK", "imagecodecs"):
        assert f'"{lib}"' in spec, lib
    assert '"nnunetv2",' not in spec.split("excludes = [")[1].split("]")[0]                # the optional feature is shipped
    assert 'collect_data_files("nnunetv2", include_py_files=True)' in spec and "dynamic_network_architectures" in spec
    for mod in ("models", "privacy", "settings", "shortcut"):
        assert f"celldform.demo.{mod}" in spec and (ROOT / "celldform" / "demo" / f"{mod}.py").is_file()
    assert "celldform/demo/assets" in spec and "THIRD_PARTY_NOTICES" in spec and "LICENSE" in spec


def test_the_requirements_are_lean_and_use_headless_opencv():
    lines = [l.split("#")[0].strip().lower() for l in read("requirements-app.txt").splitlines()]
    names = {l.split(">")[0].split("=")[0] for l in lines if l}
    assert "opencv-python-headless" in names and "opencv-python" not in names
    assert "nnunetv2" in names                                                    # optional feature: load an nnU-Net .pth
    assert not names & {"statsmodels", "napari", "torchvision", "requests", "fastapi", "mpi4py", "mkdocs", "simpleitk", "imagecodecs"}
    assert {"pyside6-essentials", "torch", "scikit-learn", "matplotlib", "pandas", "scipy", "numpy", "pyyaml", "joblib"} <= names
    assert not any(n.startswith("pyqt") for n in names), "PyQt is GPL: an installer must not contain it"


def test_the_build_script_stages_neutrally_audits_selftests_and_uploads_nothing():
    ps = read("build.ps1")
    assert "celldform_build" in ps and "/XD __pycache__ /XF *.pyc" in ps
    assert ps.count("audit_package.py") >= 3                                   # the source, the program, the installer file
    assert "--selftest" in ps and "Get-FileHash" in ps and "THIRD_PARTY_NOTICES" in ps
    for risky in ("Invoke-WebRequest", "Invoke-RestMethod", "curl ", "scp ", "ftp", "Send-MailMessage", "gh release", "twine"):
        assert risky not in ps, risky
    assert ps.index("audit_package.py") < ps.index("PyInstaller") < ps.rindex("audit_package.py")


def test_the_workflow_has_no_secrets_no_write_access_and_publishes_nothing():
    wf = (ROOT / ".github" / "workflows" / "windows-installer.yml").read_text()
    assert "permissions:\n  contents: read" in wf and "persist-credentials: false" in wf
    assert "secrets." not in wf and "GITHUB_TOKEN" not in wf
    for publish in ("gh release", "action-gh-release", "twine", "pypi", "ftp", "scp"):
        assert publish not in wf.lower(), publish
    assert "windows-latest" in wf and "upload-artifact" in wf and "retention-days" in wf


def test_the_program_entry_and_the_icon():
    assert "from celldform.demo.gui import main" in read("app_entry.py") and "freeze_support" in read("app_entry.py")
    from PIL import Image
    with Image.open(ROOT / "celldform" / "demo" / "assets" / "celldform.ico") as ico:
        sizes = ico.info.get("sizes") or {ico.size}
    assert {(16, 16), (32, 32), (256, 256)} <= set(sizes)


def test_licence_notices_state_the_lgpl_terms_for_pyside_and_nothing_gpl():
    spec2 = importlib.util.spec_from_file_location("tpn", ROOT / "installer" / "third_party_notices.py")
    tpn = importlib.util.module_from_spec(spec2); spec2.loader.exec_module(tpn)
    text = tpn.notices(["numpy", "PySide6-Essentials"])
    assert "numpy " in text and "LGPL v3" in text and "replace them" in text and "lgpl-3.0" in text and "licence:" in text
    assert "Nothing in celldform sends data anywhere" in text and "GPL v3 as well" not in text
    assert "PySide6 and the Qt libraries" not in tpn.notices(["numpy"])


def test_the_installer_recipe_never_includes_a_gpl_window_library():
    for name in ("requirements-app.txt", "celldform.spec", "build.ps1", "celldform.iss"):
        text = read(name)
        for line in text.splitlines():
            code = line.split("#")[0].lower()
            assert not ("pyqt" in code and "exclude" not in code and '"pyqt' not in code), f"{name}: {line}"
    assert '"PyQt6"' in read("celldform.spec") and "COLLECT" in read("celldform.spec")          # excluded, and a folder build


def test_the_selftest_and_version_flags(tmp_path):
    env = {**__import__("os").environ, "QT_QPA_PLATFORM": "offscreen"}
    for flag, text in (("--version", "celldform 0.1.0"), ("--selftest", "selftest OK (nnU-Net: OK)")):
        r = subprocess.run([sys.executable, "-m", "celldform.demo", flag], capture_output=True, text=True, timeout=120, cwd=ROOT, env=env)
        assert r.returncode == 0 and text in r.stdout, r.stderr[-800:]


def test_without_nnunetv2_the_app_still_starts_and_says_the_feature_is_optional(tmp_path):
    code = r'''
import sys, importlib.abc, os
class B(importlib.abc.MetaPathFinder):
    def find_spec(self, n, p, t=None):
        if n.split(".")[0] in ("nnunetv2", "SimpleITK", "batchgenerators"): raise ImportError("blocked " + n)
sys.meta_path.insert(0, B())
os.environ["QT_QPA_PLATFORM"] = "offscreen"
from celldform.demo import gui, models, segment
print("available:", segment.nnunet_available())
sys.exit(gui.selftest())
'''
    import os
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(p for p in (str(ROOT), os.environ.get("PYTHONPATH", "")) if p),
           "CELLDFORM_HOME": str(tmp_path / "h")}
    r = subprocess.run([sys.executable, "-s", "-c", code], capture_output=True, text=True, timeout=180, cwd=ROOT, env=env)
    assert r.returncode == 0 and "selftest OK (nnU-Net: not installed (optional))" in r.stdout, r.stderr[-1200:]
