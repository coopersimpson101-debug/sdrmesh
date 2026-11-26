"""Check that every relative link and image in the Markdown files resolves.

    python tools/check_links.py
"""

from __future__ import annotations

import os
import re
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PATTERNS = [re.compile(r"\]\(([^)\s]+)\)"), re.compile(r'(?:src|href)="([^"]+)"')]
SKIP_DIRS = {".git", ".pytest_cache", "build", "dist", "results-quick"}


def main() -> int:
    bad = []
    md_files = []
    for d, dirs, files in os.walk(ROOT):
        dirs[:] = [x for x in dirs if x not in SKIP_DIRS and not x.endswith(".egg-info")]
        md_files += [os.path.join(d, f) for f in files if f.endswith(".md")]
    for path in md_files:
        with open(path, encoding="utf-8") as fh:
            text = fh.read()
        for pat in PATTERNS:
            for target in pat.findall(text):
                if target.startswith(("http://", "https://", "mailto:", "#")):
                    continue
                t = target.split("#")[0]
                if not t:
                    continue
                full = os.path.normpath(os.path.join(os.path.dirname(path), t))
                if not os.path.exists(full):
                    bad.append(f"{os.path.relpath(path, ROOT)} -> {target}")
    for b in bad:
        print("broken:", b)
    print(f"checked {len(md_files)} markdown files, {len(bad)} broken links")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
