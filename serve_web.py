#!/usr/bin/env python3
"""Serve the Model Runner Hub static UI. Run from repo root: python serve_web.py [port]"""

import http.server
import os
import socketserver
import sys

DEFAULT_PORT = 8765
WEB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "web")


class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=WEB_DIR, **kwargs)


class ReuseAddressTCPServer(socketserver.TCPServer):
    allow_reuse_address = True


def main() -> None:
    env_port = os.environ.get("PORT")
    if len(sys.argv) > 1:
        preferred = int(sys.argv[1])
    elif env_port:
        preferred = int(env_port)
    else:
        preferred = DEFAULT_PORT

    last_err: OSError | None = None
    for port in range(preferred, preferred + 32):
        try:
            httpd = ReuseAddressTCPServer(("", port), Handler)
        except OSError as e:
            last_err = e
            continue
        if port != preferred:
            print(f"Port {preferred} is busy; using {port} instead.", file=sys.stderr)
        print(f"Model Runner Hub: http://127.0.0.1:{port}/")
        print("Press Ctrl+C to stop.")
        try:
            httpd.serve_forever()
        finally:
            httpd.server_close()
        return

    raise RuntimeError(
        f"Could not bind a port starting at {preferred} (last error: {last_err})"
    ) from last_err


if __name__ == "__main__":
    main()
