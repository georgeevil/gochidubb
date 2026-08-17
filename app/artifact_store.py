"""Content-addressed record of completed pipeline stages (beta).

One `artifacts` table in the same SQLite file as `jobs`: a stage fingerprint
(see app/reuse.py) mapped to the job that produced it, the files it wrote, and
the quality measured at the time. A later job whose inputs hash to the same
fingerprint can copy those files instead of recomputing them.

The table records; it never deletes. A row whose files have since been removed
is treated as a miss and pruned lazily, so cleaning up outputs/ can never
corrupt a run — the worst case is that work is done twice.

Reads and writes are synchronous and short. Every call opens its own
connection: the pipeline runs stages in a thread pool, and sqlite3 connections
are not safe to share across threads.
"""
from __future__ import annotations

import json
import logging
import os
import sqlite3
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

log = logging.getLogger("gochidubb.artifacts")

_DB_PATH: Optional[Path] = None


def init_store(db_path: Path) -> None:
    """Create the artifacts table. Call once at startup, next to init_db()."""
    global _DB_PATH
    _DB_PATH = db_path
    conn = sqlite3.connect(str(db_path))
    try:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS artifacts (
                fingerprint TEXT NOT NULL,
                stage       TEXT NOT NULL,
                job_id      TEXT NOT NULL,
                created     REAL NOT NULL,
                last_used   REAL,
                hits        INTEGER NOT NULL DEFAULT 0,
                quality     TEXT,
                payload     TEXT NOT NULL,
                PRIMARY KEY (fingerprint, stage)
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_artifacts_stage "
                     "ON artifacts(stage)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_artifacts_job "
                     "ON artifacts(job_id)")
        conn.commit()
    finally:
        conn.close()


def _connect() -> Optional[sqlite3.Connection]:
    if _DB_PATH is None:
        return None
    conn = sqlite3.connect(str(_DB_PATH), timeout=10)
    conn.row_factory = sqlite3.Row
    return conn


def record(fingerprint: str, stage: str, job_id: str, *,
           files: List[str], ctx_keys: Dict[str, Any],
           quality: Optional[Dict[str, Any]] = None) -> bool:
    """Remember that `job_id` produced this stage's output.

    `files` are paths inside the job's output directory; `ctx_keys` is the
    slice of pipeline context a reusing job needs to carry forward (segment
    lists, speaker refs, durations).

    Upsert rather than INSERT OR REPLACE: the newest producer of a
    fingerprint is the one whose files are most likely to still exist, but
    REPLACE also reset `hits` to zero, wiping the one statistic that shows
    whether the cache is doing anything.
    """
    if not fingerprint or _DB_PATH is None:
        return False
    conn = _connect()
    if conn is None:
        return False
    try:
        conn.execute(
            "INSERT INTO artifacts"
            "(fingerprint, stage, job_id, created, last_used, hits, quality, payload)"
            " VALUES(?,?,?,?,NULL,0,?,?)"
            " ON CONFLICT(fingerprint, stage) DO UPDATE SET"
            "   job_id=excluded.job_id, created=excluded.created,"
            "   quality=excluded.quality, payload=excluded.payload",
            (fingerprint, stage, job_id, time.time(),
             json.dumps(quality or {}, ensure_ascii=False),
             json.dumps({"files": files, "ctx": ctx_keys}, ensure_ascii=False)),
        )
        conn.commit()
        return True
    except sqlite3.Error as e:
        log.warning(f"[artifacts] record failed ({stage}): {e}")
        return False
    finally:
        conn.close()


def lookup(fingerprint: str, stage: str, output_root: Path) -> Optional[Dict[str, Any]]:
    """Find a reusable artifact, or None.

    Verifies every recorded file still exists before returning: outputs/ is a
    directory users delete from, and handing back a row that points at deleted
    audio would fail the run instead of merely missing the cache.
    """
    if not fingerprint or _DB_PATH is None:
        return None
    conn = _connect()
    if conn is None:
        return None
    try:
        row = conn.execute(
            "SELECT * FROM artifacts WHERE fingerprint=? AND stage=?",
            (fingerprint, stage),
        ).fetchone()
        if row is None:
            return None
        payload = json.loads(row["payload"])
        src_dir = output_root / row["job_id"]
        if not src_dir.is_dir():
            _forget(conn, fingerprint, stage)
            return None
        for rel in payload.get("files", []):
            if not (src_dir / rel).exists():
                log.info(f"[artifacts] {stage} {fingerprint[:12]} evicted — "
                         f"{rel} is gone from {row['job_id']}")
                _forget(conn, fingerprint, stage)
                return None
        return {
            "fingerprint": fingerprint,
            "stage": stage,
            "job_id": row["job_id"],
            "created": row["created"],
            "hits": row["hits"],
            "quality": json.loads(row["quality"] or "{}"),
            "files": payload.get("files", []),
            "ctx": payload.get("ctx", {}),
            "source_dir": src_dir,
        }
    except (sqlite3.Error, ValueError) as e:
        log.warning(f"[artifacts] lookup failed ({stage}): {e}")
        return None
    finally:
        conn.close()


def _forget(conn: sqlite3.Connection, fingerprint: str, stage: str) -> None:
    try:
        conn.execute("DELETE FROM artifacts WHERE fingerprint=? AND stage=?",
                     (fingerprint, stage))
        conn.commit()
    except sqlite3.Error:
        pass


def mark_hit(fingerprint: str, stage: str) -> None:
    """Count a reuse. Best-effort — a failed counter must not fail a job."""
    conn = _connect()
    if conn is None:
        return
    try:
        conn.execute(
            "UPDATE artifacts SET hits = hits + 1, last_used = ? "
            "WHERE fingerprint=? AND stage=?",
            (time.time(), fingerprint, stage),
        )
        conn.commit()
    except sqlite3.Error:
        pass
    finally:
        conn.close()


def stats(output_root: Path) -> Dict[str, Any]:
    """Per-stage summary for the beta UI."""
    conn = _connect()
    if conn is None:
        return {"enabled": False, "stages": [], "total": 0, "hits": 0}
    try:
        rows = conn.execute(
            "SELECT stage, COUNT(*) n, SUM(hits) hits, MAX(created) newest "
            "FROM artifacts GROUP BY stage ORDER BY stage"
        ).fetchall()
        stages = [{
            "stage": r["stage"], "entries": r["n"],
            "hits": r["hits"] or 0, "newest": r["newest"],
        } for r in rows]
        return {
            "enabled": True,
            "stages": stages,
            "total": sum(s["entries"] for s in stages),
            "hits": sum(s["hits"] for s in stages),
        }
    except sqlite3.Error as e:
        log.warning(f"[artifacts] stats failed: {e}")
        return {"enabled": False, "stages": [], "total": 0, "hits": 0}
    finally:
        conn.close()


def entries(stage: str = "", limit: int = 100) -> List[Dict[str, Any]]:
    """Recent artifacts, newest first, for the beta UI."""
    conn = _connect()
    if conn is None:
        return []
    try:
        sql = ("SELECT fingerprint, stage, job_id, created, last_used, hits, "
               "quality FROM artifacts")
        args: tuple = ()
        if stage:
            sql += " WHERE stage=?"
            args = (stage,)
        sql += " ORDER BY created DESC LIMIT ?"
        rows = conn.execute(sql, args + (max(1, min(limit, 500)),)).fetchall()
        return [{
            "fingerprint": r["fingerprint"], "stage": r["stage"],
            "job_id": r["job_id"], "created": r["created"],
            "last_used": r["last_used"], "hits": r["hits"],
            "quality": json.loads(r["quality"] or "{}"),
        } for r in rows]
    except (sqlite3.Error, ValueError) as e:
        log.warning(f"[artifacts] entries failed: {e}")
        return []
    finally:
        conn.close()


def purge(stage: str = "", job_id: str = "") -> int:
    """Forget artifacts. Returns how many rows were removed.

    Only ever touches this table — the files themselves belong to their jobs
    and are deleted through the normal job-deletion path.
    """
    conn = _connect()
    if conn is None:
        return 0
    try:
        sql, args = "DELETE FROM artifacts", []
        where = []
        if stage:
            where.append("stage=?")
            args.append(stage)
        if job_id:
            where.append("job_id=?")
            args.append(job_id)
        if where:
            sql += " WHERE " + " AND ".join(where)
        cur = conn.execute(sql, args)
        conn.commit()
        return cur.rowcount or 0
    except sqlite3.Error as e:
        log.warning(f"[artifacts] purge failed: {e}")
        return 0
    finally:
        conn.close()


def copy_artifacts(entry: Dict[str, Any], dest_dir: Path) -> bool:
    """Materialize a cached stage's files into a new job's directory.

    Hard-links where the filesystem allows it and copies otherwise: a reused
    transcribe stage brings a ~1 GB audio_hq.wav with it, and paying for that
    twice would eat most of what the cache saves.
    """
    import shutil

    src_dir: Path = entry["source_dir"]
    dest_dir.mkdir(parents=True, exist_ok=True)
    for rel in entry.get("files", []):
        src, dst = src_dir / rel, dest_dir / rel
        if dst.exists():
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        try:
            if src.is_dir():
                shutil.copytree(src, dst, dirs_exist_ok=True)
                continue
            try:
                os.link(src, dst)
            except OSError:
                shutil.copy2(src, dst)
        except OSError as e:
            log.warning(f"[artifacts] could not materialize {rel}: {e}")
            return False
    return True
