"""Move repository files and rewrite every relative Markdown link that pointed at them (and the moved files' own links).

    python scripts/move_docs.py moves.json [--dry]
moves.json: {"old/relative/path.md": "new/relative/path.md", ...}; plain mv (no git add), refuses to overwrite.
Script string paths are not touched: update readers separately (grep the old path).
"""
import argparse
import json
import os
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LINK = re.compile(r'(\]\()([^)\s]+)(\))')
SKIP = ('.git/', '.dvc/', '.compat/', 'node_modules/')


def main():
    p = argparse.ArgumentParser(); p.add_argument('moves'); p.add_argument('--dry', action='store_true'); a = p.parse_args()
    moves = {os.path.normpath(k): os.path.normpath(v) for k, v in json.loads(Path(a.moves).read_text()).items()}
    for old, new in moves.items():
        assert (ROOT / old).is_file() and not (ROOT / new).exists(), (old, new)
    inverse = {v: k for k, v in moves.items()}
    docs = [p.relative_to(ROOT).as_posix() for p in ROOT.rglob('*.md') if not str(p.relative_to(ROOT)).startswith(SKIP)
            and '/artifacts/' not in str(p) and not p.is_symlink()]
    edits = {}
    for doc in docs:
        here_old = os.path.dirname(doc); here_new = os.path.dirname(moves.get(doc, doc))
        text = (ROOT / doc).read_text(); changed = 0

        def fix(m):
            nonlocal changed
            target = m.group(2)
            if re.match(r'^[a-z][a-z0-9+.-]*:', target) or target.startswith(('#', '/')): return m.group(0)
            path, _, anchor = target.partition('#')
            if not path: return m.group(0)
            resolved = os.path.normpath(os.path.join(here_old, path))
            dest = moves.get(resolved, resolved)
            if dest == resolved and here_old == here_new: return m.group(0)
            if not (ROOT / dest).exists() and resolved not in moves: return m.group(0)   # leave already-broken links alone
            rel = os.path.relpath(dest, here_new) + (('#' + anchor) if anchor else '')
            if rel != target: changed += 1
            return m.group(1) + rel + m.group(3)
        new_text = LINK.sub(fix, text)
        if changed: edits[doc] = (new_text, changed)
    print('moves', len(moves), 'docs with rewritten links', len(edits), 'links', sum(c for _, c in edits.values()))
    if a.dry: return
    for doc, (text, _) in edits.items(): (ROOT / doc).write_text(text)
    for old, new in moves.items():
        (ROOT / new).parent.mkdir(parents=True, exist_ok=True); (ROOT / old).rename(ROOT / new)


if __name__ == '__main__':
    main()
