"""WebSocket endpoint for sequence update notifications."""
from __future__ import annotations

from typing import Dict, List

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

router = APIRouter(prefix="/ws", tags=["websocket"])

_subscribers: Dict[str, List[WebSocket]] = {}


async def _subscribe(sequence_id: str, websocket: WebSocket) -> None:
    await websocket.accept()
    bucket = _subscribers.setdefault(sequence_id, [])
    bucket.append(websocket)


async def _unsubscribe(sequence_id: str, websocket: WebSocket) -> None:
    bucket = _subscribers.get(sequence_id, [])
    if websocket in bucket:
        bucket.remove(websocket)


async def broadcast(sequence_id: str, message: dict) -> None:
    # _subscribers accumulates connections across every reload/tab for this
    # sequence_id (shared identically across actors) -- a single stale/dead
    # socket that disconnected without a clean WebSocketDisconnect (network
    # blip, tab reload, etc.) would otherwise abort send_json() and silently
    # stop delivery to every subscriber later in the list, even genuinely live
    # ones. Isolate each send so one dead socket can't swallow the rest, and
    # drop it from the bucket so it doesn't keep failing on every future event.
    bucket = _subscribers.get(sequence_id, [])
    dead: List[WebSocket] = []
    for socket in list(bucket):
        try:
            await socket.send_json(message)
        except Exception:
            dead.append(socket)
    for socket in dead:
        if socket in bucket:
            bucket.remove(socket)


@router.websocket("/sequences/{sequence_id}")
async def websocket_sequence_updates(sequence_id: str, websocket: WebSocket) -> None:
    await _subscribe(sequence_id, websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        await _unsubscribe(sequence_id, websocket)
