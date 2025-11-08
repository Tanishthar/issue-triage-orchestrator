# packages/ws/manager.py
import asyncio
import json
from typing import Set
from starlette.websockets import WebSocket

# Global singleton instance will be created when imported by the app
class WSManager:
    def __init__(self):
        self.active: Set[WebSocket] = set()
        # Lock for modifying active set
        self._lock = asyncio.Lock()

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        async with self._lock:
            self.active.add(websocket)

    async def disconnect(self, websocket: WebSocket):
        async with self._lock:
            if websocket in self.active:
                self.active.remove(websocket)

    async def send_json(self, websocket: WebSocket, payload: dict):
        try:
            await websocket.send_text(json.dumps(payload))
        except Exception:
            # If send fails, ensure we remove socket from list (caller may also remove)
            await self.disconnect(websocket)

    async def broadcast(self, payload: dict):
        """
        Send payload (dict) to all connected clients.
        If a websocket errors, remove it.
        """
        if not self.active:
            return

        # make a stable list snapshot to iterate
        async with self._lock:
            sockets = list(self.active)

        coros = []
        for ws in sockets:
            coros.append(self.send_json(ws, payload))

        # run concurrently but don't crash on individual failures
        await asyncio.gather(*coros, return_exceptions=True)


# Export a single manager instance to import elsewhere
manager = WSManager()


# Helper to schedule an async broadcast from sync code:
def notify_async(payload: dict):
    """
    Safe helper for sync contexts: schedule manager.broadcast(payload)
    in the running event loop.
    """
    try:
        loop = asyncio.get_running_loop()
        # if running, schedule task directly
        loop.create_task(manager.broadcast(payload))
    except RuntimeError:
        # No running loop (rare in startup phases) — try to get loop and call soon-threadsafe
        try:
            loop = asyncio.get_event_loop()
            loop.call_soon_threadsafe(lambda: asyncio.create_task(manager.broadcast(payload)))
        except Exception:
            # if there's absolutely no loop, we just drop notification (local dev; logs still persist)
            pass
