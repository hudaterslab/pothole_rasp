"""Run with the application's Python environment: python -m unittest -v."""
from contextlib import contextmanager
from datetime import datetime, timedelta
import json
from pathlib import Path
import socket
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

import numpy as np

import run_main as app


def nmea(body):
    checksum = 0
    for character in body:
        checksum ^= ord(character)
    return f"${body}*{checksum:02X}\r\n".encode("ascii")


GGA = nmea("GNGGA,123519,3723.2475,N,12702.1234,E,4,12,0.8,45.2,M,0.0,M,,")
RMC = nmea("GNRMC,123519,A,3723.2475,N,12702.1234,E,2.5,84.4,071026,,,A")


class GpsTcpTests(unittest.TestCase):
    def wait_for(self, condition):
        deadline = time.monotonic() + 4.0
        while not condition() and time.monotonic() < deadline:
            time.sleep(0.01)
        self.assertTrue(condition(), "GNSS receiver did not reach the expected state")

    @contextmanager
    def receiver(self, serve):
        finished = threading.Event()
        errors = []
        with socket.socket() as listener, tempfile.TemporaryDirectory() as root:
            listener.bind(("127.0.0.1", 0))
            listener.listen()
            listener.settimeout(5.0)
            session = app.GpsSession(datetime.now(), root)
            recorder = app.GpsNmeaRecorder(
                f"tcp://127.0.0.1:{listener.getsockname()[1]}", 115200, session,
            )

            def run_server():
                try:
                    serve(listener, finished)
                except Exception as exc:
                    if not finished.is_set():
                        errors.append(exc)

            server = threading.Thread(target=run_server, daemon=True)
            server.start()
            with patch.object(app, "GPS_RECONNECT_INITIAL_SEC", 0.01):
                recorder.start()
                try:
                    yield recorder, session
                finally:
                    finished.set()
                    recorder.stop()
                    server.join(timeout=6.0)
            self.assertFalse(recorder.thread.is_alive())
            self.assertFalse(server.is_alive())
            self.assertEqual(errors, [])
            self.assertFalse(recorder.is_connected())

    def test_fragmented_tcp_sentences_reach_image_metadata(self):
        partial_sent = threading.Event()
        send_rest = threading.Event()

        def serve(listener, finished):
            with listener.accept()[0] as connection:
                connection.sendall(GGA[:17])
                partial_sent.set()
                send_rest.wait(timeout=3.0)
                connection.sendall(GGA[17:] + RMC)
                finished.wait(timeout=5.0)

        with self.receiver(serve) as (recorder, session):
            self.assertTrue(partial_sent.wait(timeout=3.0))
            self.assertEqual(recorder.stats()["sentence_count"], 0)
            send_rest.set()
            self.wait_for(lambda: recorder.stats()["valid_fix_count"] == 2)
            self.wait_for(lambda: session.gps_jsonl.read_text().count("\n") == 2)
            rows = [json.loads(line) for line in session.gps_jsonl.read_text().splitlines()]
            self.assertEqual([row["sentence_type"] for row in rows], ["GGA", "RMC"])
            self.assertEqual(rows[0]["fix_quality"], 4)
            self.assertEqual(rows[0]["satellites"], 12)
            self.assertAlmostEqual(rows[0]["altitude_m"], 45.2)
            self.assertAlmostEqual(rows[1]["speed_knots"], 2.5)
            self.assertAlmostEqual(rows[1]["course_deg"], 84.4)
            self.assertTrue(recorder.is_connected())
            self.assertEqual(recorder.stats()["transport"], "tcp")

            streams = session.gps_tail.read()
            timestamp = rows[1]["timestamp"]
            app.save_detection_data(
                np.zeros((20, 30, 3), dtype=np.uint8), [], 15,
                root_dir=session.root_dir, frame_timestamp=timestamp, gps_streams=streams,
            )
            metadata = list(Path(session.root_dir).glob("*/*/meta/*.json"))
            self.assertEqual(len(metadata), 1)
            gps = json.loads(metadata[0].read_text())["gps"]
            self.assertAlmostEqual(gps["latitude_deg"], 37 + 23.2475 / 60)
            self.assertAlmostEqual(gps["longitude_deg"], 127 + 2.1234 / 60)
            self.assertEqual(app.get_current_gps_fix(timestamp + 2.0, streams), (None, None))

    def test_invalid_fix_and_checksum_never_supply_a_position(self):
        invalid = nmea("GNRMC,123519,V,3723.2475,N,12702.1234,E,2.5,84.4,071026,,,N")
        no_fix = nmea("GNGGA,123519,3723.2475,N,12702.1234,E,0,00,9.9,45.2,M,0.0,M,,")
        corrupt = RMC[:-4] + b"ZZ\r\n"

        def serve(listener, finished):
            with listener.accept()[0] as connection:
                connection.sendall(invalid + no_fix + corrupt)
                finished.wait(timeout=5.0)

        with self.receiver(serve) as (recorder, session):
            self.wait_for(lambda: recorder.stats()["sentence_count"] == 3)
            self.wait_for(lambda: session.gps_jsonl.read_text().count("\n") == 3)
            self.assertEqual(recorder.stats()["valid_fix_count"], 0)
            self.assertEqual(recorder.stats()["checksum_error_count"], 1)
            streams = session.gps_tail.read()
            timestamp = recorder.stats()["last_sentence_timestamp"]
            self.assertEqual(app.get_current_gps_fix(timestamp, streams), (None, None))

    def test_disconnect_discards_partial_sentence_before_reconnecting(self):
        southwest = nmea("GNRMC,123519,A,3723.2475,S,12702.1234,W,2.5,84.4,071026,,,A")

        def serve(listener, finished):
            with listener.accept()[0] as first:
                first.sendall(GGA[:17])
            with listener.accept()[0] as second:
                second.sendall(southwest)
                finished.wait(timeout=5.0)

        with self.receiver(serve) as (recorder, session):
            self.wait_for(lambda: recorder.stats()["valid_fix_count"] == 1)
            stats = recorder.stats()
            self.assertEqual(stats["receiver_open_count"], 2)
            self.assertEqual(stats["sentence_count"], 1)
            self.assertEqual(stats["checksum_error_count"], 0)
            self.assertAlmostEqual(stats["latest_valid_fix"]["latitude_deg"], -(37 + 23.2475 / 60))
            self.assertAlmostEqual(stats["latest_valid_fix"]["longitude_deg"], -(127 + 2.1234 / 60))

    def test_silent_tcp_connection_is_reopened(self):
        def serve(listener, finished):
            with listener.accept()[0] as first:
                first.settimeout(3.0)
                self.assertEqual(first.recv(1), b"")
            with listener.accept()[0] as second:
                second.sendall(GGA)
                finished.wait(timeout=5.0)

        with patch.object(app, "GPS_TCP_IDLE_TIMEOUT_SEC", 0.05):
            with self.receiver(serve) as (recorder, session):
                self.wait_for(lambda: recorder.stats()["valid_fix_count"] == 1)
                self.assertEqual(recorder.stats()["receiver_open_count"], 2)
                self.assertTrue(any("stopped sending data" in error
                                    for error in recorder.stats()["transport_errors"]))

    def test_connection_failure_retries_and_recovers(self):
        connect = socket.create_connection
        attempts = []

        def connect_after_failure(*args, **kwargs):
            attempts.append(args)
            if len(attempts) == 1:
                raise ConnectionRefusedError("receiver not ready")
            return connect(*args, **kwargs)

        def serve(listener, finished):
            with listener.accept()[0] as connection:
                connection.sendall(RMC)
                finished.wait(timeout=5.0)

        with patch.object(app.socket, "create_connection", side_effect=connect_after_failure):
            with self.receiver(serve) as (recorder, session):
                self.wait_for(lambda: recorder.stats()["valid_fix_count"] == 1)
                self.assertEqual(len(attempts), 2)
                self.assertTrue(any("receiver not ready" in error
                                    for error in recorder.stats()["transport_errors"]))

    def test_tcp_log_rotates_without_reconnecting(self):
        send_second = threading.Event()

        def serve(listener, finished):
            with listener.accept()[0] as connection:
                connection.sendall(GGA)
                send_second.wait(timeout=3.0)
                connection.sendall(RMC)
                finished.wait(timeout=5.0)

        with self.receiver(serve) as (recorder, session):
            self.wait_for(lambda: session.gps_jsonl.exists() and
                          session.gps_jsonl.read_text().count("\n") == 1)
            next_session = app.GpsSession(datetime.now() + timedelta(minutes=10), session.root_dir)
            recorder.rebind_recorder(next_session, 0.0)
            send_second.set()
            self.wait_for(lambda: next_session.gps_jsonl.exists() and
                          next_session.gps_jsonl.read_text().count("\n") == 1)
            self.assertEqual(json.loads(session.gps_jsonl.read_text())["sentence_type"], "GGA")
            self.assertEqual(json.loads(next_session.gps_jsonl.read_text())["sentence_type"], "RMC")
            self.assertIs(recorder.recorder, next_session)
            self.assertEqual(recorder.stats()["receiver_open_count"], 1)


if __name__ == "__main__":
    unittest.main()
