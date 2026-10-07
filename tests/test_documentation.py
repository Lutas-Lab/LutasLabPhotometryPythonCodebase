import re
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DOCUMENTATION_ROOTS = (
    REPOSITORY_ROOT / "README.md",
    REPOSITORY_ROOT / "docs",
    REPOSITORY_ROOT / "notebooks" / "README.md",
    REPOSITORY_ROOT / "lutaslab_photometry" / "README.md",
    REPOSITORY_ROOT / "packages",
)


def markdown_files():
    for root in DOCUMENTATION_ROOTS:
        if root.is_file():
            yield root
        else:
            yield from root.rglob("*.md")


def shell_fence_lines(path: Path):
    shell = None
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        marker = line.strip().lower()
        if shell is None and marker in {"```powershell", "```bash"}:
            shell = marker.removeprefix("```")
            continue
        if shell is not None and marker == "```":
            shell = None
            continue
        if shell is not None:
            yield shell, line_number, line


def test_shell_blocks_use_matching_line_continuations():
    errors = []
    for path in markdown_files():
        for shell, line_number, line in shell_fence_lines(path):
            if shell == "powershell" and line.endswith("\\"):
                errors.append(f"{path.relative_to(REPOSITORY_ROOT)}:{line_number}: bash \\")
            if shell == "bash" and line.endswith("`"):
                errors.append(
                    f"{path.relative_to(REPOSITORY_ROOT)}:{line_number}: PowerShell `"
                )
            if shell == "powershell" and line.rstrip().endswith("`") and not line.endswith("`"):
                errors.append(
                    f"{path.relative_to(REPOSITORY_ROOT)}:{line_number}: "
                    "characters follow the PowerShell continuation"
                )

    assert not errors, "Mismatched shell continuation syntax:\n" + "\n".join(errors)


def test_local_markdown_links_resolve():
    errors = []
    pattern = re.compile(r"!?\[[^\]]*\]\(([^)]+)\)")
    for path in markdown_files():
        for target in pattern.findall(path.read_text(encoding="utf-8")):
            target = target.strip("<>").split("#", 1)[0]
            if not target or target.startswith(("http://", "https://", "mailto:")):
                continue
            if not (path.parent / target).exists():
                errors.append(f"{path.relative_to(REPOSITORY_ROOT)}: {target}")

    assert not errors, "Broken local documentation links:\n" + "\n".join(errors)
