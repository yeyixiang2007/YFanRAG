#!/usr/bin/env python
"""Release helper: update version/changelog and optionally create git tag."""

from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path
import re
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
PYPROJECT = ROOT / "pyproject.toml"
CHANGELOG = ROOT / "CHANGELOG.md"


def main() -> int:
    args = parse_args()
    version = args.version.strip()
    validate_version(version)

    if not args.skip_tests:
        run(["python", "-m", "pytest", "-q"], cwd=ROOT)

    update_pyproject_version(version, dry_run=args.dry_run)
    update_changelog(version, dry_run=args.dry_run)

    if args.tag:
        run(["git", "tag", "-a", f"v{version}", "-m", f"Release v{version}"], cwd=ROOT, dry_run=args.dry_run)

    print(f"release prepared for v{version}")
    if args.dry_run:
        print("dry-run enabled: no files or tags were changed")
    else:
        print("next steps:")
        print("1. git add pyproject.toml CHANGELOG.md")
        print(f"2. git commit -m \"Release v{version}\"")
        if not args.tag:
            print(f"3. git tag -a v{version} -m \"Release v{version}\"")
        print("4. git push && git push --tags")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare a project release")
    parser.add_argument("version", help="target version, e.g. 0.1.0")
    parser.add_argument("--tag", action="store_true", help="create git tag automatically")
    parser.add_argument("--skip-tests", action="store_true", help="skip running tests")
    parser.add_argument("--dry-run", action="store_true", help="preview changes without writing")
    return parser.parse_args()


def validate_version(version: str) -> None:
    if not re.match(r"^\d+\.\d+\.\d+$", version):
        raise ValueError(f"invalid version: {version}; expected SemVer X.Y.Z")


def update_pyproject_version(version: str, dry_run: bool) -> None:
    content = PYPROJECT.read_text(encoding="utf-8")
    updated = re.sub(
        r'(?m)^version = ".*"$',
        f'version = "{version}"',
        content,
        count=1,
    )
    if content == updated:
        raise RuntimeError("failed to update version in pyproject.toml")
    if not dry_run:
        PYPROJECT.write_text(updated, encoding="utf-8")


def update_changelog(version: str, dry_run: bool) -> None:
    if CHANGELOG.exists():
        content = CHANGELOG.read_text(encoding="utf-8")
    else:
        content = "# Changelog\n\n## [Unreleased]\n\n"

    release_header = f"## [{version}] - {date.today().isoformat()}"
    if release_header in content:
        return

    pattern = re.compile(r"(?m)^## \[Unreleased\]\s*$")
    match = pattern.search(content)
    if not match:
        content += "\n## [Unreleased]\n\n"
        match = pattern.search(content)
        if not match:
            raise RuntimeError("failed to find or create [Unreleased] section")

    insert_at = match.end()
    addition = (
        "\n\n"
        f"{release_header}\n\n"
        "### Added\n\n"
        "- TODO: summarize release changes.\n"
    )
    updated = content[:insert_at] + addition + content[insert_at:]
    if not dry_run:
        CHANGELOG.write_text(updated, encoding="utf-8")


def run(cmd: list[str], cwd: Path, dry_run: bool = False) -> None:
    if dry_run:
        print("dry-run:", " ".join(cmd))
        return
    subprocess.run(cmd, cwd=str(cwd), check=True)


if __name__ == "__main__":
    raise SystemExit(main())
