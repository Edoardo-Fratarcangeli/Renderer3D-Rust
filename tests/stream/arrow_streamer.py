#!/usr/bin/env python3
"""Arrow IPC test streamer: connect to the app (TCP server) and push a batch.

Usage: arrow_streamer.py HOST PORT N_ROWS N_COLS [N_BATCHES]

Writes an Arrow IPC *stream* (schema + record batches + EOS) over the socket.
Feature columns ``f0..f{d-1}`` hold value ``i*d + k`` for row ``i``; a ``label``
string column holds ``c{i % 3}`` — matching the NDJSON streamer so both wire
formats can be asserted identically.
"""
import socket
import sys

import pyarrow as pa


def make_batch(d: int, base: int, n: int) -> pa.RecordBatch:
    cols = {}
    for k in range(d):
        cols["f%d" % k] = pa.array(
            [float((base + i) * d + k) for i in range(n)], type=pa.float64()
        )
    cols["label"] = pa.array(["c%d" % ((base + i) % 3) for i in range(n)], type=pa.string())
    return pa.record_batch(cols)


def main() -> int:
    host = sys.argv[1]
    port = int(sys.argv[2])
    n = int(sys.argv[3])
    d = int(sys.argv[4])
    n_batches = int(sys.argv[5]) if len(sys.argv) > 5 else 1

    sock = socket.create_connection((host, port))
    sink = sock.makefile("wb")
    try:
        schema = make_batch(d, 0, 1).schema
        writer = pa.ipc.new_stream(sink, schema)
        per = max(1, n // n_batches)
        sent = 0
        for _ in range(n_batches):
            count = min(per, n - sent)
            if count <= 0:
                break
            writer.write_batch(make_batch(d, sent, count))
            sent += count
        writer.close()
        sink.flush()
    finally:
        try:
            sock.shutdown(socket.SHUT_WR)
        except OSError:
            pass
        sink.close()
        sock.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
