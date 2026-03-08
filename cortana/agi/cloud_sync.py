"""
Cloud Memory Persistence — backup and restore Cortana's SQLite state.

On Render (and other ephemeral hosts), the local filesystem is wiped
on each deploy/restart. This module syncs the SQLite database to a
cloud storage backend so memory, goals, world model, and consciousness
state survive across deploys.

Supported backends (configured via .env):
  CLOUD_SYNC_BACKEND = supabase | s3 | r2 | http | none (default)

Supabase Storage (recommended free tier):
  SUPABASE_URL      = https://xxxx.supabase.co
  SUPABASE_KEY      = <service role key>
  SUPABASE_BUCKET   = cortana-memory

AWS S3 / Cloudflare R2:
  CLOUD_S3_ENDPOINT = <endpoint url>
  CLOUD_S3_BUCKET   = <bucket name>
  CLOUD_S3_KEY      = <access key id>
  CLOUD_S3_SECRET   = <secret access key>

HTTP (generic presigned URL pair):
  CLOUD_UPLOAD_URL  = PUT URL
  CLOUD_DOWNLOAD_URL = GET URL

Sync schedule:
  - On startup: attempt restore from cloud
  - Every CLOUD_SYNC_INTERVAL_MINUTES (default 15): upload snapshot
  - On clean shutdown: final upload
"""
from __future__ import annotations

import io
import os
import shutil
import sqlite3
import threading
import time
from pathlib import Path
from typing import Optional

from cortana import config

_DB_PATH       = Path(config.SQLITE_PATH)
_BACKEND       = os.getenv("CLOUD_SYNC_BACKEND", "none").lower()
_INTERVAL_SECS = int(os.getenv("CLOUD_SYNC_INTERVAL_MINUTES", "15")) * 60
_OBJECT_KEY    = "cortana_memory.sqlite"


# ---------------------------------------------------------------------------
# Backend implementations
# ---------------------------------------------------------------------------

def _upload_supabase(data: bytes) -> bool:
    import requests
    url = os.getenv("SUPABASE_URL", "").rstrip("/")
    key = os.getenv("SUPABASE_KEY", "")
    bucket = os.getenv("SUPABASE_BUCKET", "cortana-memory")
    if not url or not key:
        return False
    endpoint = f"{url}/storage/v1/object/{bucket}/{_OBJECT_KEY}"
    headers  = {"Authorization": f"Bearer {key}", "Content-Type": "application/octet-stream"}
    r = requests.put(endpoint, data=data, headers=headers, timeout=30)
    return r.status_code in (200, 201)


def _download_supabase() -> Optional[bytes]:
    import requests
    url = os.getenv("SUPABASE_URL", "").rstrip("/")
    key = os.getenv("SUPABASE_KEY", "")
    bucket = os.getenv("SUPABASE_BUCKET", "cortana-memory")
    if not url or not key:
        return None
    endpoint = f"{url}/storage/v1/object/{bucket}/{_OBJECT_KEY}"
    headers  = {"Authorization": f"Bearer {key}"}
    r = requests.get(endpoint, headers=headers, timeout=30)
    if r.status_code == 200:
        return r.content
    return None


def _upload_s3(data: bytes) -> bool:
    try:
        import boto3
        endpoint = os.getenv("CLOUD_S3_ENDPOINT")
        bucket   = os.getenv("CLOUD_S3_BUCKET", "cortana")
        key_id   = os.getenv("CLOUD_S3_KEY", "")
        secret   = os.getenv("CLOUD_S3_SECRET", "")
        kwargs   = dict(
            aws_access_key_id=key_id,
            aws_secret_access_key=secret,
        )
        if endpoint:
            kwargs["endpoint_url"] = endpoint
        s3 = boto3.client("s3", **kwargs)
        s3.put_object(Bucket=bucket, Key=_OBJECT_KEY, Body=data)
        return True
    except Exception:
        return False


def _download_s3() -> Optional[bytes]:
    try:
        import boto3
        endpoint = os.getenv("CLOUD_S3_ENDPOINT")
        bucket   = os.getenv("CLOUD_S3_BUCKET", "cortana")
        key_id   = os.getenv("CLOUD_S3_KEY", "")
        secret   = os.getenv("CLOUD_S3_SECRET", "")
        kwargs   = dict(
            aws_access_key_id=key_id,
            aws_secret_access_key=secret,
        )
        if endpoint:
            kwargs["endpoint_url"] = endpoint
        s3 = boto3.client("s3", **kwargs)
        obj = s3.get_object(Bucket=bucket, Key=_OBJECT_KEY)
        return obj["Body"].read()
    except Exception:
        return None


def _upload_http(data: bytes) -> bool:
    import requests
    upload_url = os.getenv("CLOUD_UPLOAD_URL", "")
    if not upload_url:
        return False
    r = requests.put(upload_url, data=data, timeout=30)
    return r.status_code < 300


def _download_http() -> Optional[bytes]:
    import requests
    download_url = os.getenv("CLOUD_DOWNLOAD_URL", "")
    if not download_url:
        return None
    r = requests.get(download_url, timeout=30)
    if r.status_code == 200:
        return r.content
    return None


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

def _snapshot_db() -> bytes:
    """Create a consistent binary snapshot of the SQLite database."""
    buf = io.BytesIO()
    src = sqlite3.connect(str(_DB_PATH))
    dst = sqlite3.connect(":memory:")
    src.backup(dst)
    dst.backup(sqlite3.connect(str(_DB_PATH)))  # flush WAL
    src.close()
    # Re-read as raw bytes
    return _DB_PATH.read_bytes()


def upload() -> bool:
    """Upload current DB to cloud. Returns True on success."""
    if _BACKEND == "none" or not _DB_PATH.exists():
        return False
    try:
        data = _snapshot_db()
        if _BACKEND == "supabase":
            return _upload_supabase(data)
        if _BACKEND in ("s3", "r2"):
            return _upload_s3(data)
        if _BACKEND == "http":
            return _upload_http(data)
    except Exception:
        pass
    return False


def restore() -> bool:
    """Download DB from cloud and restore. Returns True on success."""
    if _BACKEND == "none":
        return False
    try:
        data: Optional[bytes] = None
        if _BACKEND == "supabase":
            data = _download_supabase()
        elif _BACKEND in ("s3", "r2"):
            data = _download_s3()
        elif _BACKEND == "http":
            data = _download_http()

        if data and len(data) > 1024:
            # Backup existing before overwriting
            if _DB_PATH.exists():
                shutil.copy2(_DB_PATH, str(_DB_PATH) + ".bak")
            _DB_PATH.write_bytes(data)
            return True
    except Exception:
        pass
    return False


# ---------------------------------------------------------------------------
# Background sync thread
# ---------------------------------------------------------------------------

class CloudSyncDaemon:
    """Periodic upload daemon — runs as a daemon thread."""

    def __init__(self) -> None:
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._last_upload = 0.0

    def start(self) -> None:
        if _BACKEND == "none":
            return
        self._running = True
        self._thread = threading.Thread(
            target=self._loop,
            name="cortana-cloud-sync",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        # Final upload on clean shutdown
        try:
            upload()
        except Exception:
            pass

    def _loop(self) -> None:
        while self._running:
            time.sleep(60)
            if time.time() - self._last_upload >= _INTERVAL_SECS:
                try:
                    ok = upload()
                    self._last_upload = time.time()
                except Exception:
                    pass


# Module-level singleton
_daemon = CloudSyncDaemon()


def start_sync() -> None:
    """Call once at startup (after restore()) to begin periodic uploads."""
    _daemon.start()


def stop_sync() -> None:
    """Call on clean shutdown — does one final upload."""
    _daemon.stop()
