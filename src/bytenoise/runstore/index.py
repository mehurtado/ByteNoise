"""sqlite cache over run directories. Rebuildable at any time; run files stay the source of truth."""
import json
import sqlite3

from ..core.paths import cache_dir, runs_dir
from .reader import MARKERS, RunInfo, describe_run, payload_dir


def _key(path):
    """Cheap change detector: marker file mtimes and sizes, at the run root and in its trainer payload dir."""
    parts = []
    for base in {path, payload_dir(path)}:
        for m in MARKERS:
            f = base / m
            if f.exists():
                st = f.stat()
                parts.append(f"{base.name}/{m}:{st.st_mtime_ns}:{st.st_size}")
        ck = base / "checkpoint.pt"
        if ck.exists():
            parts.append(f"{base.name}/checkpoint.pt:{ck.stat().st_mtime_ns}")
    return "|".join(sorted(parts))


_VOLATILE = ("running", "created", "unknown")  # status depends on the clock, so never trust the cache for these


def _connect():
    db = sqlite3.connect(cache_dir() / "runs_index.db", timeout=30)
    try:
        db.execute("create table if not exists runs (id text primary key, key text, info text)")
    except Exception:
        db.close()  # release the file handle so a corrupt cache can be deleted on Windows
        raise
    return db


def _open():
    """Open the cache; a corrupt file is deleted and rebuilt (it is only a cache)."""
    try:
        return _connect()
    except sqlite3.DatabaseError:
        (cache_dir() / "runs_index.db").unlink(missing_ok=True)
        return _connect()


def refresh():
    """Re-describe only runs whose files changed (or whose status is clock-dependent); drop vanished ones.
    Returns (infos, stats)."""
    root = runs_dir()
    db = _open()
    try:
        have = {r[0]: (r[1], r[2]) for r in db.execute("select id, key, info from runs")}
        infos, seen, parsed, skipped = [], set(), 0, 0
        dirs = [p for p in sorted(root.iterdir()) if p.is_dir()] if root.exists() else []
        for p in dirs:
            seen.add(p.name)
            k = _key(p)
            if not k:
                skipped += 1
                continue
            cached = have.get(p.name)
            if cached and cached[0] == k:
                try:
                    info = RunInfo(**json.loads(cached[1]))
                    if info.status not in _VOLATILE:
                        infos.append(info)
                        continue
                except (TypeError, ValueError):
                    pass  # schema drifted: re-describe
            try:
                info = describe_run(p)
            except Exception:
                skipped += 1
                continue
            if info is None:
                skipped += 1
                continue
            parsed += 1
            infos.append(info)
            db.execute("insert or replace into runs values (?,?,?)", (p.name, k, json.dumps(info.to_dict())))
        for gone in set(have) - seen:
            db.execute("delete from runs where id=?", (gone,))
        db.commit()
    finally:
        db.close()
    return infos, {"runs": len(infos), "reparsed": parsed, "skipped": skipped}
