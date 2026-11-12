"""The freeze record (contract §1.4): nothing is scored on the test or
prospective period before the method is frozen.

A freeze is declared by the author, by committing a file `FREEZE` at the
repository root with one line, `tag: <git tag>`, and creating that tag on the
frozen commit. `require` lets a command touch a period other than validation
only if the file is there, the tag exists, and `src/` is unchanged since the
tag, so test numbers always come from the frozen code. There is no override.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

FREEZE_FILE = Path("FREEZE")
SOURCE_DIR = "src"


class NotFrozen(RuntimeError):
    pass


def _git(root: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True)


def frozen_tag(root: Path = Path(".")) -> str:
    """The tag named in the freeze record, after checking it; raises NotFrozen with the reason."""
    path = Path(root) / FREEZE_FILE
    if not path.exists():
        raise NotFrozen(f"the method is not frozen (no {FREEZE_FILE} file)")
    m = re.search(r"^tag:\s*(\S+)\s*$", path.read_text(), flags=re.M)
    if not m:
        raise NotFrozen(f"{FREEZE_FILE} has no `tag: <git tag>` line")
    tag = m.group(1)
    try:
        if _git(root, "rev-parse", "-q", "--verify", f"refs/tags/{tag}").returncode != 0:
            raise NotFrozen(f"{FREEZE_FILE} names the tag {tag}, which does not exist in this clone")
        if _git(root, "diff", "--quiet", tag, "--", SOURCE_DIR).returncode != 0:
            raise NotFrozen(f"{SOURCE_DIR}/ differs from the frozen tag {tag}")
    except FileNotFoundError:
        raise NotFrozen("git is not available, so the freeze tag cannot be checked") from None
    return tag


def require(period: str, root: Path = Path(".")) -> None:
    """Raise NotFrozen unless `period` is validation or the method is frozen."""
    if period == "validation":
        return
    try:
        frozen_tag(root)
    except NotFrozen as err:
        raise NotFrozen(f"{period} period refused: {err}. Only the validation period is scored before "
                        "the freeze (contract §1.4).") from None
