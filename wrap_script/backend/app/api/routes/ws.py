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
    for socket in list(_subscribers.get(sequence_id, [])):
        await socket.send_json(message)


@router.websocket("/sequences/{sequence_id}")
async def websocket_sequence_updates(sequence_id: str, websocket: WebSocket) -> None:
    await _subscribe(sequence_id, websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        await _unsubscribe(sequence_id, websocket)
