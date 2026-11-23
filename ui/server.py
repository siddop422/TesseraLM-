"""Loopback-only browser interface for TesseraLM inference."""
from __future__ import annotations
import argparse
import json
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from inference.generate import load_model, generate_tokens
from model.gpt import count_parameters

ROOT = Path(__file__).resolve().parents[1]
MODEL = TOKENIZER = CHECKPOINT = DEVICE = None
INFERENCE_LOCK = threading.Lock()


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        print("[web] " + fmt % args)

    def send_body(self, status, content_type, body):
        data = body if isinstance(body, bytes) else body.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Referrer-Policy", "no-referrer")
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            page = ROOT / "ui" / "index.html"
            self.send_body(200, "text/html; charset=utf-8", page.read_text(encoding="utf-8"))
        elif self.path == "/health":
            info = {"status": "ok", "model": getattr(MODEL, "model_name", "TesseraLM-Nano"),
                    "device": DEVICE, "parameters": count_parameters(MODEL),
                    "checkpoint": CHECKPOINT.name}
            self.send_body(200, "application/json; charset=utf-8", json.dumps(info))
        else:
            self.send_body(404, "text/plain; charset=utf-8", "Not found")

    def do_POST(self):
        if self.path != "/api/chat":
            self.send_body(404, "application/json; charset=utf-8", '{"error":"Not found"}')
            return
        if not INFERENCE_LOCK.acquire(blocking=False):
            self.send_body(429, "application/json; charset=utf-8",
                           json.dumps({"error": "The model is generating another reply. Try again shortly."}))
            return
        try:
            size = int(self.headers.get("Content-Length", "0"))
            if size < 1 or size > 20000:
                raise ValueError("Request body must be between 1 and 20000 bytes")
            data = json.loads(self.rfile.read(size))
            prompt = str(data.get("prompt", ""))[-8000:]
            if not prompt.strip():
                raise ValueError("Enter a prompt first")
            settings = data.get("settings", {})
            maximum = max(1, min(200, int(settings.get("max_new_tokens", 80))))
            temperature = max(0.1, min(2.0, float(settings.get("temperature", 0.85))))
            top_k = max(0, min(100, int(settings.get("top_k", 40))))
            top_p = max(0.1, min(1.0, float(settings.get("top_p", 0.9))))
            penalty = max(1.0, min(2.0, float(settings.get("repetition_penalty", 1.08))))
            prompt_ids = TOKENIZER.encode(prompt)
        except (ValueError, TypeError, json.JSONDecodeError) as exc:
            INFERENCE_LOCK.release()
            self.send_body(400, "application/json; charset=utf-8", json.dumps({"error": str(exc)}))
            return

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache, no-transform")
        self.send_header("Connection", "close")
        self.send_header("X-Accel-Buffering", "no")
        self.end_headers()
        generated = []
        try:
            for token in generate_tokens(MODEL, prompt_ids, maximum, temperature, top_k, top_p, penalty):
                generated.append(token)
                event = json.dumps({"text": TOKENIZER.decode(generated), "done": False}, ensure_ascii=False)
                self.wfile.write(("data: " + event + "\n\n").encode("utf-8"))
                self.wfile.flush()
            event = json.dumps({"text": TOKENIZER.decode(generated), "done": True,
                                "tokens": len(generated)}, ensure_ascii=False)
            self.wfile.write(("data: " + event + "\n\n").encode("utf-8"))
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception as exc:
            try:
                event = json.dumps({"error": str(exc), "done": True})
                self.wfile.write(("data: " + event + "\n\n").encode("utf-8"))
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass
        finally:
            INFERENCE_LOCK.release()
            self.close_connection = True


def main():
    global MODEL, TOKENIZER, CHECKPOINT, DEVICE
    parser = argparse.ArgumentParser(description="Run TesseraLM local browser chat")
    parser.add_argument("--checkpoint", default="checkpoints/nano.pt")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()
    CHECKPOINT = (ROOT / args.checkpoint).resolve() if not Path(args.checkpoint).is_absolute() else Path(args.checkpoint)
    print("Loading checkpoint:", CHECKPOINT)
    MODEL, TOKENIZER, meta, DEVICE = load_model(CHECKPOINT, "cpu")
    MODEL.model_name = meta.get("model_config", {}).get("name", "TesseraLM-Nano")
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    server.daemon_threads = True
    url = "http://{}:{}".format(args.host, args.port)
    print("TesseraLM ready at", url, "| device:", DEVICE)
    if not args.no_browser:
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping TesseraLM...")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
