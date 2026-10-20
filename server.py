"""LayaMed: local server (FastAPI) that serves web/index.html and answers the case analysis.

Usage (in the layamed environment):
    python server.py          this machine only: http://127.0.0.1:8000
    python server.py --lan    also for other computers on the same local network: http://<this machine's IP>:8000
In --lan mode the server only accepts private-network addresses (home, lab); never open the port to the
internet. Requests must also name this machine in the Host header (127.0.0.1, localhost or, with --lan, its network
IPs and name), which blocks DNS rebinding from a web page; LAYAMED_ALLOWED_HOSTS replaces that list (comma-separated,
"*" for any, e.g. behind the Colab proxy). Clients see only the probabilities: the weights and the training data
never leave this machine.
The Laya logic lives in layamed/service.py; this file only holds the API and the page.
"""
import ipaddress
import os
import socket
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))  # allows "import layamed" without installing the package

import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import FileResponse, PlainTextResponse
from pydantic import BaseModel, Field

from layamed import inference as inf
from layamed.service import EXAMPLE_LABELS, EXAMPLES, LABELS, Service

PORT = int(os.environ.get("PORT", 8000))  # another port: $env:PORT = "8001" before running
LAN = "--lan" in sys.argv
MAX_CHARS = 60_000  # single field; a clinical case rarely exceeds 10 thousand characters

SERVICE = Service(ROOT)
app = FastAPI(title="LayaMed", docs_url=None, redoc_url=None, openapi_url=None)


def lan_ips():
    """Private IPv4 addresses of this machine (the ones other computers on the network use to reach it)."""
    try:
        ips = {e[4][0] for e in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET)}
    except socket.gaierror:
        return []
    return sorted(ip for ip in ips if ipaddress.ip_address(ip).is_private and not ipaddress.ip_address(ip).is_loopback)


def allowed_hosts():
    if os.environ.get("LAYAMED_ALLOWED_HOSTS"):
        return [h.strip() for h in os.environ["LAYAMED_ALLOWED_HOSTS"].split(",") if h.strip()]
    hosts = ["127.0.0.1", "localhost"]
    if LAN:
        name = socket.gethostname()
        hosts += lan_ips() + [name, name + ".local", name.lower(), name.lower() + ".local"]
    return list(dict.fromkeys(hosts))  # without repeats


app.add_middleware(TrustedHostMiddleware, allowed_hosts=allowed_hosts())


@app.middleware("http")
async def local_network_only(request: Request, call_next):
    # Refuses any client outside a private network (e.g. a port forwarded on the router by mistake)
    try:
        ip = ipaddress.ip_address(request.client.host)
        allowed = ip.is_private or ip.is_loopback
    except (AttributeError, ValueError):
        allowed = False
    if not allowed:
        return PlainTextResponse("LayaMed: local network only.", status_code=403)
    return await call_next(request)


class Case(BaseModel):
    text: str = Field("", max_length=MAX_CHARS)  # single field; sections come from the titles the user types


def example_text(hpi, pe, lab):
    # Example case in the single-field format: each section with its title (empty sections are left out)
    parts = [("History of present illness", hpi), ("Physical examination", pe), ("Labs", lab)]
    return "\n\n".join(f"{t}:\n{c.strip()}" for t, c in parts if c.strip())


@app.get("/")
def page():
    return FileResponse(ROOT / "web" / "index.html", headers={"Cache-Control": "no-store"})


@app.get("/logo.svg")
def logo():
    return FileResponse(ROOT / "assets" / "laya_logo.svg", media_type="image/svg+xml")


@app.get("/api/config")
def config():
    """Fixed interface data: classes, labels, example cases and limits."""
    return {"classes": inf.CLASSES, "labels": LABELS, "max_len": inf.MAX_LEN, "mask": inf.MASK_TOKEN,
            "max_chars": MAX_CHARS,
            "examples": [{"label": lb, "text": example_text(*ex)} for lb, ex in zip(EXAMPLE_LABELS, EXAMPLES)]}


@app.post("/api/analyze")
def analyze(case: Case):
    # Synchronous on purpose: FastAPI runs it in a worker thread and the Service serializes GPU access
    sections = inf.split_free_text(case.text)
    result = SERVICE.analyze(sections["hpi"], sections["pe"], sections["lab"])
    result["warnings"]["inline_title"] = inf.inline_titles(case.text)
    return {**result, "sections_read": [s for s in inf.SECTIONS if sections[s]]}


if __name__ == "__main__":
    print(f"LayaMed at http://127.0.0.1:{PORT}")
    if LAN:
        for ip in lan_ips() or ["(no local network IP found)"]:
            print(f"  other computers on the network: http://{ip}:{PORT}")
    uvicorn.run(app, host="0.0.0.0" if LAN else "127.0.0.1", port=PORT, log_level="warning")
