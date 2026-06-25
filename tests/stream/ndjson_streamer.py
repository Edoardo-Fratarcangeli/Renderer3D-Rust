#!/usr/bin/env python3
"""NDJSON test streamer: connect to the app (TCP server) and push rows.

Usage: ndjson_streamer.py HOST PORT N_ROWS N_COLS [DELAY_SECONDS]

Each row is a JSON object ``{"x": [f0, f1, ...], "label": "cK"}`` followed by a
newline. Row ``i`` carries features ``i*d + k`` and label ``c{i % 3}`` so the
consumer can verify the decoded values deterministically.
"""
import json
import socket
import sys
import time


def main() -> int:
    host = sys.argv[1]
    port = int(sys.argv[2])
    n = int(sys.argv[3])
    d = int(sys.argv[4])
    delay = float(sys.argv[5]) if len(sys.argv) > 5 else 0.0

    sock = socket.create_connection((host, port))
    try:
        for i in range(n):
            row = {"x": [float(i * d + k) for k in range(d)], "label": "c%d" % (i % 3)}
            sock.sendall((json.dumps(row) + "\n").encode("utf-8"))
            if delay:
                time.sleep(delay)
    finally:
        try:
            sock.shutdown(socket.SHUT_WR)
        except OSError:
            pass
        sock.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
