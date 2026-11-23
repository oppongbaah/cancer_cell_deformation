"""
Write the licence notices for everything bundled into the installer.

    python installer/third_party_notices.py > THIRD_PARTY_NOTICES.txt           # every package in this environment
    python installer/third_party_notices.py --only numpy PySide6-Essentials     # just these

Run it inside the build environment, after the requirements are installed, so the list is what is really shipped.
PySide6 is LGPL v3: shipping it asks that the Qt libraries stay replaceable files, which the folder build does.
"""
from __future__ import annotations

import argparse
import sys
from importlib import metadata
from typing import Iterable, List, Optional

HEADER = """celldform - third-party software notices
==========================================

celldform itself is MIT licensed (see LICENSE). The installer also contains the open-source packages below, each under
its own licence. Nothing in celldform sends data anywhere; these notices are about licences only.
"""
PYSIDE = """
PySide6 and the Qt libraries
----------------------------
The window system of this program is PySide6 (Qt for Python, The Qt Company), licensed under the GNU LGPL v3 (it is also
available under the GPL v3 and a commercial licence), built on Qt, which is licensed under the LGPL v3 / GPL v3.
celldform uses them as separate dynamic libraries: in the installation folder the Qt and PySide6 files (.dll, .pyd) are
ordinary files, not packed into the program, so you may replace them with another build of the same version. The full
text of the LGPL v3 is at https://www.gnu.org/licenses/lgpl-3.0.txt and the source of Qt and PySide6 at
https://code.qt.io/ and https://pypi.org/project/PySide6/ . celldform itself is MIT licensed.
"""


def licence_of(dist: metadata.Distribution) -> str:
    md = dist.metadata
    expr = md.get("License-Expression")
    if expr:
        return expr
    lic = (md.get("License") or "").strip()
    if lic and len(lic) < 120 and "\n" not in lic:
        return lic
    classifiers = [c.split("::")[-1].strip() for c in md.get_all("Classifier") or [] if c.startswith("License ::")]
    return "; ".join(classifiers) or (lic.splitlines()[0][:100] if lic else "see the package")


def notices(only: Optional[Iterable[str]] = None) -> str:
    want = {o.lower().replace("_", "-") for o in only} if only else None
    rows: List[str] = []
    for dist in sorted(metadata.distributions(), key=lambda d: (d.metadata.get("Name") or "").lower()):
        name = dist.metadata.get("Name")
        if not name or (want is not None and name.lower().replace("_", "-") not in want):
            continue
        home = dist.metadata.get("Home-page") or next((u.split(",")[-1].strip() for u in dist.metadata.get_all("Project-URL") or []), "")
        rows.append(f"{name} {dist.version}\n    licence: {licence_of(dist)}" + (f"\n    {home}" if home.startswith("http") else ""))
    return HEADER + "\n" + "\n\n".join(rows) + "\n" + (PYSIDE if want is None or want & {"pyside6", "pyside6-essentials", "shiboken6"} else "")


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--only", nargs="*", default=None)
    a = ap.parse_args(argv)
    sys.stdout.write(notices(a.only))
    return 0


if __name__ == "__main__":
    sys.exit(main())
