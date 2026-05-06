"""WebSocket server for live SUMO V2V frame streaming.

Exposes:
  GET /health          — liveness probe
  WS  /ws/live         — broadcast channel; clients receive one JSON message
                         per simulation step, identical in shape to latest.json
  GET /live/*          — static serve of data/live/
  GET /frames/*        — static serve of data/frames/
  GET /fused/*         — static serve of data/fused/
  GET /images/*        — static serve of data/images/
  GET /sumo_map.net.xml — current SUMO network file

Run standalone (for testing):
  uvicorn ws_server:app --host 0.0.0.0 --port 8765 --reload

Used by run_live_pipeline.py which calls start_background_server() at startup
and push_frame() once per simulation step.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import threading
import time
from pathlib import Path
from typing import Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

logger = logging.getLogger(__name__)

app = FastAPI(title="SUMO V2V WebSocket Server")

# Allow browser connections from any origin (Vite dev server, direct, etc.)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Connection registry
# ---------------------------------------------------------------------------

_connected: set[WebSocket] = set()
_lock = threading.Lock()


# ---------------------------------------------------------------------------
# Startup signal — fired by uvicorn when it is actually ready to accept
# connections.  Used by start_background_server() to know when the port is
# open before returning to the caller.
# ---------------------------------------------------------------------------


@app.on_event("startup")
async def _on_uvicorn_startup() -> None:
    _server_started.set()


# ---------------------------------------------------------------------------
# WebSocket endpoint
# ---------------------------------------------------------------------------


@app.get("/health")
async def health() -> JSONResponse:
    with _lock:
        n = len(_connected)
    return JSONResponse({"status": "ok", "clients": n})


@app.websocket("/ws/live")
async def live_ws(ws: WebSocket) -> None:
    await ws.accept()
    with _lock:
        _connected.add(ws)
    client = ws.client
    logger.info("[ws] Client connected: %s", client)
    try:
        while True:
            # Keep-alive: wait for any client message (ping / close).
            await ws.receive_text()
    except WebSocketDisconnect:
        logger.info("[ws] Client disconnected: %s", client)
    except Exception as exc:  # pragma: no cover
        logger.warning("[ws] Connection error (%s): %s", client, exc)
    finally:
        with _lock:
            _connected.discard(ws)


# ---------------------------------------------------------------------------
# Static data endpoints — mounted dynamically by start_background_server()
# ---------------------------------------------------------------------------

_data_root: Path | None = None


def _mount_static_dirs(data_root: Path) -> None:
    """Mount simulation output directories as static file routes.

    Called once by start_background_server() before uvicorn starts so the
    directories exist before the first request comes in.
    """
    global _data_root
    _data_root = data_root

    routes = {
        "/live":   "live",
        "/frames": "frames",
        "/fused":  "fused",
        "/images": "images",
    }
    for route, subdir in routes.items():
        path = data_root / subdir
        path.mkdir(parents=True, exist_ok=True)
        # Use a unique name to avoid duplicate-mount errors on re-use.
        app.mount(route, StaticFiles(directory=str(path), html=False), name=subdir)
        logger.info("[ws] Serving %s → %s", route, path)


@app.get("/sumo_map.net.xml")
async def serve_sumo_map() -> FileResponse:
    """Serve the current SUMO network file so the frontend can parse it."""
    if _data_root is None:
        return FileResponse(status_code=404, path=__file__)  # fallback
    net_file = _data_root / "sumo_map.net.xml"
    if not net_file.exists():
        return FileResponse(status_code=404, path=__file__)
    return FileResponse(str(net_file), media_type="application/xml")


@app.get("/live/latest.json")
async def serve_latest_json() -> JSONResponse:
    """Serve latest.json with a safe read to avoid mid-write hangs."""
    if _data_root is None:
        return JSONResponse({}, status_code=404)
    latest_file = _data_root / "live" / "latest.json"
    if not latest_file.exists():
        return JSONResponse({}, status_code=404)

    last_err: Exception | None = None
    for _ in range(3):
        try:
            text = latest_file.read_text(encoding="utf-8")
            if not text.strip():
                return JSONResponse({}, status_code=204)
            payload = json.loads(text)
            return JSONResponse(payload)
        except Exception as exc:  # pragma: no cover
            last_err = exc
            time.sleep(0.01)

    logger.warning("[ws] latest.json read failed: %s", last_err)
    return JSONResponse({}, status_code=204)


# ---------------------------------------------------------------------------
# Broadcast helper (called from synchronous pipeline thread via asyncio bridge)
# ---------------------------------------------------------------------------


async def broadcast(payload: dict[str, Any]) -> None:
    """Send *payload* as JSON text to every connected WebSocket client."""
    msg = json.dumps(payload)
    dead: set[WebSocket] = set()

    with _lock:
        targets = set(_connected)

    for ws in targets:
        try:
            await ws.send_text(msg)
        except Exception as exc:
            logger.debug("[ws] Send failed, dropping client: %s", exc)
            dead.add(ws)

    if dead:
        with _lock:
            _connected -= dead


# ---------------------------------------------------------------------------
# Background server launcher (called from run_live_pipeline.py)
# ---------------------------------------------------------------------------

_server_loop: asyncio.AbstractEventLoop | None = None
_server_started = threading.Event()


def start_background_server(
    host: str = "0.0.0.0",
    port: int = 8765,
    data_root: Path | None = None,
) -> None:
    """Spin up uvicorn in a daemon thread.

    Blocks until the server is ready (up to ~10 s) so callers can safely
    call push_frame() immediately afterwards.
    """
    if data_root is not None:
        _mount_static_dirs(data_root)

    def _run() -> None:
        global _server_loop
        import uvicorn  # imported here so the module loads without uvicorn if unused

        _server_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(_server_loop)

        config = uvicorn.Config(
            app,
            host=host,
            port=port,
            loop="none",
            log_level="warning",
            access_log=False,
        )
        server = uvicorn.Server(config)

        async def _serve() -> None:
            # _server_started is set by the @app.on_event("startup") hook
            # once uvicorn has bound the port and is accepting connections.
            await server.serve()

        _server_loop.run_until_complete(_serve())

    t = threading.Thread(target=_run, name="ws-server", daemon=True)
    t.start()
    ok = _server_started.wait(timeout=10.0)
    if ok:
        print(f"[ws] WebSocket server ready -> ws://{host}:{port}/ws/live")
        print(f"[ws] Static data server    -> http://{host}:{port}/frames|live|fused|images")
    else:
        print(f"[ws] WARNING: WebSocket server did not start within 10 s on port {port}")


def push_frame(payload: dict[str, Any]) -> None:
    """Thread-safe fire-and-forget broadcast from the synchronous pipeline."""
    if _server_loop is None or _server_loop.is_closed():
        return
    asyncio.run_coroutine_threadsafe(broadcast(payload), _server_loop)
