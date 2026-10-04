"""OpenAI-compatible HTTP server using only the stdlib.

Endpoints:
- GET  /health                    -> {"status": "ok"}
- GET  /v1/models                 -> model list
- POST /v1/chat/completions       -> OpenAI chat completions (with honesty gating)
- POST /v1/completions            -> legacy completions

No third-party web framework: a small `http.server` handler routes JSON. This
keeps the whole framework importable in a sandboxed / managed Python.
"""
import json
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

from ..config import Config, load_config
from ..utils.logging import get_logger

log = get_logger("kolibriforge.serving")


def _build_app(cfg):
    from ..engine import KolibriEngine
    engine = KolibriEngine(cfg)
    return engine


def make_handler(engine):
    class Handler(BaseHTTPRequestHandler):
        def _send(self, code, obj):
            body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _read_json(self):
            length = int(self.headers.get("Content-Length", 0))
            if length == 0:
                return {}
            return json.loads(self.rfile.read(length).decode("utf-8"))

        def do_GET(self):
            if self.path == "/health":
                self._send(200, {"status": "ok"})
            elif self.path == "/v1/models":
                self._send(200, {"object": "list", "data": [
                    {"id": engine.cfg.backend.model, "object": "model",
                     "owned_by": "kolibriforge"}]})
            else:
                self._send(404, {"error": "not found"})

        def do_POST(self):
            if self.path in ("/v1/chat/completions", "/v1/completions"):
                try:
                    payload = self._read_json()
                    prompt = _extract_prompt(payload, self.path)
                    reasoning_level = payload.get("reasoning_level") or payload.get(
                        "kolibri_reasoning_level")
                    verdict = engine.answer(prompt, reasoning_level=reasoning_level)
                    content = verdict["answer"] or ""
                    finish = "stop"
                    if verdict["action"] != "accept":
                        finish = "abstain"
                    resp = {
                        "id": f"chatcmpl-{int(time.time() * 1000)}",
                        "object": "chat.completion",
                        "created": int(time.time()),
                        "model": engine.cfg.backend.model,
                        "choices": [{
                            "index": 0,
                            "message": {"role": "assistant", "content": content},
                            "finish_reason": finish,
                        }],
                        "kolibri": verdict,
                    }
                    self._send(200, resp)
                except Exception as e:
                    log.error("completions error: %s", e)
                    self._send(500, {"error": str(e)})
            else:
                self._send(404, {"error": "not found"})

        def log_message(self, *args):
            pass  # keep stdout clean

    return Handler


def _extract_prompt(payload, path):
    if path == "/v1/chat/completions":
        msgs = payload.get("messages", [])
        return "\n".join(m.get("content", "") for m in msgs if isinstance(m, dict))
    return payload.get("prompt", "")


def app(cfg: Config = None):
    cfg = cfg or load_config()
    engine = _build_app(cfg)
    return make_handler(engine)


def serve(cfg=None, host="127.0.0.1", port=8000):
    cfg = cfg or load_config()
    engine = _build_app(cfg)
    handler = make_handler(engine)
    server = HTTPServer((host, port), handler)
    log.info("kolibriforge server listening on http://%s:%d (OpenAI-compatible)", host, port)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
