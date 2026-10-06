"""Run with the application's Python environment: python -m unittest -v."""
import io
from pathlib import Path
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

import run_main as app


DETECTION = [10, 20, 110, 120, 0.8, "Sewer_Road", "빗물받이", 4]


class FrameAlignmentTests(unittest.TestCase):
    def test_resize_and_coordinates_use_the_same_original(self):
        original = np.full((720, 1280, 3), 37, dtype=np.uint8)
        pothole = Mock()
        roadobj = Mock()
        pothole.infer.return_value = [DETECTION]
        roadobj.infer.return_value = []

        detections = app.infer_main_frame(pothole, roadobj, original)

        resized = pothole.infer.call_args.args[0]
        self.assertEqual(resized.shape, (480, 640, 3))
        self.assertTrue(np.all(resized == 37))
        self.assertIs(roadobj.infer.call_args.args[0], resized)
        self.assertEqual(detections[0][:4], [20, 30, 220, 180])
        self.assertTrue(np.all(original == 37))

    def test_newer_capture_during_inference_cannot_replace_result_frame(self):
        source = app.LatestItemBuffer()
        results = app.LatestItemBuffer()
        stop = threading.Event()
        first = np.full((1080, 1920, 3), 31, dtype=np.uint8)
        newer = np.full_like(first, 201)
        source.put((15, first, 1000.25))

        def infer_slowly(image, classes):
            self.assertTrue(np.all(image == 31))
            source.put((30, newer, 1000.75))
            stop.set()
            return [DETECTION]

        pothole = Mock()
        roadobj = Mock()
        pothole.infer.side_effect = infer_slowly
        roadobj.infer.return_value = []
        uploads = Mock()
        app.ai_worker_loop(pothole, roadobj, source, results, stop, uploads)

        frame_id, original, timestamp, detections = results.get_and_clear()
        self.assertEqual((frame_id, timestamp), (15, 1000.25))
        self.assertIs(original, first)
        self.assertEqual(detections[0][:4], [30, 45, 330, 270])
        self.assertTrue(np.all(roadobj.infer.call_args.args[0] == 31))
        self.assertIsNone(results.get_and_clear())
        self.assertIs(source.get_and_clear()[1], newer)

    def test_reader_keeps_source_id_and_receive_time_at_sampling_boundary(self):
        raw = bytes([128] * 24)  # 4x4 NV12
        packets = Mock()
        pipe = Mock()
        reader = app.FFmpegStreamReader("rtsp://test", 4, 4, packets)
        with patch.object(app.subprocess, "Popen", return_value=pipe), \
             patch.object(app, "read_exact", side_effect=[raw] * 30 + [None]), \
             patch.object(app.time, "time", side_effect=range(1001, 1031)):
            reader.run()

        self.assertEqual(packets.put.call_count, 2)
        first, second = [call.args[0] for call in packets.put.call_args_list]
        self.assertEqual((first[0], first[2]), (15, 1015))
        self.assertEqual((second[0], second[2]), (30, 1030))
        self.assertEqual(first[1].shape, (4, 4, 3))
        self.assertIsNot(first[1], second[1])
        pipe.terminate.assert_called_once()

    def test_main_saves_each_result_with_its_own_pixels_id_and_time(self):
        stop = threading.Event()
        source = Mock()
        results = Mock()
        originals = [np.full((1080, 1920, 3), value, dtype=np.uint8) for value in (21, 189)]
        packets = iter([(15, originals[0], 1000.25, [DETECTION]),
                        (30, originals[1], 1000.75, [DETECTION]),
                        (45, originals[1], 1001.25, [])])

        def next_result():
            try:
                return next(packets)
            except StopIteration:
                stop.set()
                return None

        results.get_and_clear.side_effect = next_result
        gps = Mock()
        gps.thread = None
        gps.stats.return_value = dict(active_device=None, sentence_count=0,
                                      valid_fix_count=0, transport_errors=[])
        session = SimpleNamespace(gps_jsonl=Path("/unused/session/gps/gps.jsonl"))
        uploads = Mock()
        uploads.lock = threading.RLock()

        with patch.object(app, "LatestItemBuffer", side_effect=[source, results]), \
             patch.object(app.threading, "Event", return_value=stop), \
             patch.object(app.threading, "Thread"), \
             patch.object(app.signal, "signal"), \
             patch.object(app, "FFmpegStreamReader"), \
             patch.object(app, "DeepXPPUModel"), \
             patch.object(app, "SessionUploads", return_value=uploads), \
             patch.object(app, "GpsSession", return_value=session), \
             patch.object(app, "GpsNmeaRecorder", return_value=gps), \
             patch.object(app, "refresh_gps_session", return_value=session), \
             patch.object(app, "save_detection_data", return_value=session) as save, \
             patch.object(app, "SHOW_WINDOW", False), \
             patch("sys.stdout", new_callable=io.StringIO):
            app.main()

        self.assertEqual(save.call_count, 2)
        for call, expected, frame_id, timestamp in zip(
                save.call_args_list, originals, (15, 30), (1000.25, 1000.75)):
            np.testing.assert_array_equal(call.args[0], expected)
            self.assertEqual(call.args[2], frame_id)
            self.assertEqual(call.kwargs["frame_timestamp"], timestamp)
        source.get_and_clear.assert_not_called()
        results.get_current.assert_not_called()


if __name__ == "__main__":
    unittest.main()
