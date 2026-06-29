import json
import asyncio
import urllib.request
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request, Response
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

app = FastAPI()
templates = Jinja2Templates(directory="app/templates")

MEDIAMTX_WHEP = "http://localhost:8889"


class TelemetryManager:
    def __init__(self):
        self.active_connections = []
        self.target_color = "all"

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

    async def broadcast(self, message: str):
        dead = []
        for ws in self.active_connections:
            try:
                await ws.send_text(message)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.disconnect(ws)


manager = TelemetryManager()


@app.get("/", response_class=HTMLResponse)
async def get_dashboard(request: Request):
    return templates.TemplateResponse(request=request, name="index.html", context={"request": request})


def _do_whep_request(url: str, sdp_body: bytes):
    """Blocking WHEP POST — runs in a thread executor."""
    req = urllib.request.Request(
        url,
        data=sdp_body,
        headers={"Content-Type": "application/sdp"},
        method="POST",
    )
    try:
        resp = urllib.request.urlopen(req, timeout=10)
        return resp.status, resp.read()
    except urllib.request.HTTPError as e:
        return e.code, e.read()
    except Exception:
        return 503, b"MediaMTX unreachable"


@app.post("/whep/{path:path}")
async def whep_proxy(path: str, request: Request):
    body = await request.body()
    url = f"{MEDIAMTX_WHEP}/{path}/whep"
    loop = asyncio.get_event_loop()
    status, content = await loop.run_in_executor(None, _do_whep_request, url, body)
    return Response(content=content, status_code=status, media_type="application/sdp")


@app.websocket("/ws/telemetry")
async def websocket_telemetry(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)
            if "target_color" in message:
                manager.target_color = message["target_color"]
    except WebSocketDisconnect:
        manager.disconnect(websocket)
