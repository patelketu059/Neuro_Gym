from __future__ import annotations
import json
from pathlib import Path

CHECKPOINT_KEYS = (
    'embedded_pages',
    'embedded_text',
    'embedded_tables'
)

def load(path: Path) -> dict[str, set]:
    if path.is_file():
        try:
            raw = json.loads(path.read_text(encoding = 'utf-8'))
            return {k: set(raw.get(k, [])) for k in CHECKPOINT_KEYS}
        except (json.JSONDecodeError, Exception) as e:
            print(f"[CHECKPOINT] - Error {e}")
    return {k: set() for k in CHECKPOINT_KEYS}

def save(
        path: Path,
        state: dict[str, set]
) -> None:
    
    path.parent.mkdir(parents = True, exist_ok = True)
    serialisable = {k: sorted(state.get(k, set())) for k in CHECKPOINT_KEYS}
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(serialisable, indent = 2), encoding = "utf-8")
    tmp.replace(path)

def mark_done(state: dict[str, set],
              key: str,
              item_id: str) -> None:
    state[key].add(item_id)


def is_done(state: dict[str, set],
            key: str,
            item_id: str) -> bool:
    return item_id in state[key]


def progress(
        state: dict[str, set],
        key: str,
        total: int) -> str:
    
    done = len(state.get(key, set()))
    pct  = done / total * 100 if total > 0 else 0.0
    return f"{done:,}/{total:,}  ({pct:.1f}%)"


def summary(state: dict[str, set]) -> str:
    lines = ["Checkpoint state:"]
    for k in CHECKPOINT_KEYS:
        lines.append(f"  {k}: {len(state.get(k, set())):,} done")
    return "\n".join(lines)