import os
import cv2
import numpy as np
import subprocess
import threading
import time
import json
from datetime import datetime, timezone, timedelta
import math
import select
from glob import glob
from pathlib import Path
from typing import Optional
import queue
from dx_engine import InferenceEngine, InferenceOption

# --- [1] 타겟 클래스 매핑 (2개 모델 분리 및 클래스 한정) ---
# 형식: YOLO_Index: ('JSON_영문명', '화면_출력용_한글명', 데이터_명세_Category_ID)

# 1. pothole_best_ppu.dxnn 모델 (빗물받이 전용)
POTHOLE_CLASSES = {
    3: ('Sewer_Road', '빗물받이', 4)
}

# 2. roadobj_ppu.dxnn 모델 (도로 시설물 5종 전용)
ROADOBJ_CLASSES = {
    3: ('Road_Mirror', '도로반사경', 7),
    4: ('Speed_Bump', '과속방지턱', 5),
    5: ('Traffic_Signal', '교통신호기', 3),
    8: ('Street_Name_Plate', '도로명판', 2),
    10: ('CCTV', '감시카메라(CCTV)', 6)
}

# --- [2] 비동기 데이터 공유용 버퍼 클래스 ---
class LatestItemBuffer:
    def __init__(self):
        self.item = None
        self.has_new = False
        self.lock = threading.Lock()

    def put(self, item):
        with self.lock:
            self.item = item
            self.has_new = True

    def get_and_clear(self):
        with self.lock:
            if not self.has_new:
                return None
            self.has_new = False
            return self.item

    def get_current(self):
        with self.lock:
            return self.item

# --- [3] USB serial GNSS (pothole/live_detection d8bfb3c8) ---
GPS_DEVICE = "auto"
GPS_PREFERRED_DEVICE = "/dev/serial/by-id/usb-1a86_USB_Serial-if00-port0"
GPS_BAUDRATE = 115200
GPS_RECONNECT_INITIAL_SEC = 1.0
GPS_RECONNECT_MAX_SEC = 30.0
GPS_SYNC_SEC = 2.0
lidar__GPS_MAX_TIME_ERROR_SEC = 1.5

def _nmea_float(value: str):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None

def _nmea_coordinate(value: str, hemisphere: str):
    """Convert NMEA ddmm.mmmm/dddmm.mmmm coordinates to signed degrees."""
    numeric = _nmea_float(value)
    if numeric is None:
        return None
    degrees = int(numeric // 100)
    minutes = numeric - degrees * 100
    if minutes < 0 or minutes >= 60:
        return None
    result = degrees + minutes / 60.0
    direction = str(hemisphere or "").upper()
    if direction in {"S", "W"}:
        result = -result
    elif direction not in {"N", "E"}:
        return None
    return result

def parse_nmea_sentence(raw_line: str, received_timestamp: float) -> dict:
    """Parse one NMEA sentence without inventing a location when fix is absent."""
    raw = str(raw_line).strip()
    base = {
        "timestamp": float(received_timestamp),
        "raw": raw,
        "checksum_valid": False,
        "sentence_type": None,
        "talker": None,
        "valid": False,
        "latitude_deg": None,
        "longitude_deg": None,
    }
    if not raw.startswith("$") or "*" not in raw:
        base["parse_error"] = "missing_nmea_envelope"
        return base
    body, checksum_text = raw[1:].rsplit("*", 1)
    if len(checksum_text) < 2:
        base["parse_error"] = "missing_checksum"
        return base
    checksum = 0
    for character in body:
        checksum ^= ord(character)
    try:
        expected = int(checksum_text[:2], 16)
    except ValueError:
        base["parse_error"] = "invalid_checksum_text"
        return base
    base["checksum_valid"] = checksum == expected
    fields = body.split(",")
    message_id = fields[0] if fields else ""
    if len(message_id) >= 5:
        base["talker"] = message_id[:2]
        base["sentence_type"] = message_id[-3:]
    if not base["checksum_valid"]:
        base["parse_error"] = "checksum_mismatch"
        return base

    sentence_type = base["sentence_type"]
    try:
        if sentence_type == "GGA":
            latitude = _nmea_coordinate(fields[2], fields[3])
            longitude = _nmea_coordinate(fields[4], fields[5])
            fix_quality = int(fields[6] or 0)
            base.update(
                {
                    "gps_utc_time": fields[1] or None,
                    "latitude_deg": latitude,
                    "longitude_deg": longitude,
                    "fix_quality": fix_quality,
                    "satellites": int(fields[7] or 0),
                    "hdop": _nmea_float(fields[8]),
                    "altitude_m": _nmea_float(fields[9]),
                    "valid": bool(
                        fix_quality > 0 and latitude is not None and longitude is not None
                    ),
                }
            )
        elif sentence_type == "RMC":
            latitude = _nmea_coordinate(fields[3], fields[4])
            longitude = _nmea_coordinate(fields[5], fields[6])
            status = fields[2].upper() if len(fields) > 2 else "V"
            speed_knots = _nmea_float(fields[7])
            base.update(
                {
                    "gps_utc_time": fields[1] or None,
                    "gps_status": status,
                    "latitude_deg": latitude,
                    "longitude_deg": longitude,
                    "speed_knots": speed_knots,
                    "speed_mps": (None if speed_knots is None else speed_knots * 0.514444),
                    "course_deg": _nmea_float(fields[8]),
                    "gps_date_ddmmyy": fields[9] or None,
                    "valid": bool(status == "A" and latitude is not None and longitude is not None),
                }
            )
        elif sentence_type == "GLL":
            latitude = _nmea_coordinate(fields[1], fields[2])
            longitude = _nmea_coordinate(fields[3], fields[4])
            status = fields[6].upper() if len(fields) > 6 else "V"
            base.update(
                {
                    "gps_utc_time": fields[5] or None,
                    "gps_status": status,
                    "latitude_deg": latitude,
                    "longitude_deg": longitude,
                    "valid": bool(status == "A" and latitude is not None and longitude is not None),
                }
            )
        elif sentence_type == "VTG":
            base.update(
                {
                    "course_deg": _nmea_float(fields[1]),
                    "speed_knots": _nmea_float(fields[5]),
                    "speed_kmh": _nmea_float(fields[7]),
                }
            )
        elif sentence_type == "GSA":
            base.update(
                {
                    "fix_type": int(fields[2] or 1),
                    "pdop": _nmea_float(fields[15]) if len(fields) > 15 else None,
                    "hdop": _nmea_float(fields[16]) if len(fields) > 16 else None,
                    "vdop": _nmea_float(fields[17]) if len(fields) > 17 else None,
                }
            )
    except (IndexError, ValueError) as exc:
        base["parse_error"] = f"malformed_{sentence_type or 'unknown'}:{exc}"
        base["valid"] = False
    return base

def discover_gps_serial_devices(
    device_setting: str = "auto",
    preferred_device: str = "",
) -> list[str]:
    """Return distinct GPS serial candidates, preferring stable by-id paths.

    ``auto`` first selects the configured stable preferred device when it is
    present.  Otherwise it intentionally refuses to guess when more than one
    USB serial device exists.  The recorder retries discovery, so unplug/replug
    and ttyUSB number changes do not require restarting camera/LiDAR collection.
    """
    setting = str(device_setting or "auto").strip()
    if setting.lower() != "auto":
        matches = sorted(glob(setting)) if any(ch in setting for ch in "*?[") else [setting]
        return [str(Path(path)) for path in matches]

    candidates = []
    seen_targets = set()
    for pattern in (
        "/dev/serial/by-id/*",
        "/dev/ttyACM*",
        "/dev/ttyUSB*",
    ):
        for path in sorted(glob(pattern)):
            try:
                target = os.path.realpath(path)
            except OSError:
                continue
            if not target or target in seen_targets:
                continue
            seen_targets.add(target)
            candidates.append(str(Path(path)))

    preferred = str(preferred_device or "").strip()
    if preferred:
        preferred_paths = (
            sorted(glob(preferred)) if any(ch in preferred for ch in "*?[") else [preferred]
        )
        preferred_targets = {
            os.path.realpath(path) for path in preferred_paths if os.path.exists(path)
        }
        preferred_candidates = [
            path for path in candidates if os.path.realpath(path) in preferred_targets
        ]
        if len(preferred_candidates) == 1:
            return preferred_candidates
    return candidates

class GpsNmeaRecorder:
    """Read USB NMEA asynchronously so GPS latency cannot block camera/LiDAR."""

    def __init__(
        self,
        device: str,
        baudrate: int,
        recorder: "RunRecorder",
        preferred_device: str = "",
    ):
        self.device = str(device)
        self.preferred_device = str(preferred_device)
        self.active_device = None
        self.candidate_devices = []
        self.baudrate = int(baudrate)
        self.recorder = recorder
        self.stop_event = threading.Event()
        self.thread = None
        self.lock = threading.Lock()
        self.fd = None
        self.sentence_count = 0
        self.checksum_error_count = 0
        self.parse_error_count = 0
        self.valid_fix_count = 0
        self.receiver_open_count = 0
        self.last_sentence_timestamp = None
        self.last_valid_fix_timestamp = None
        self.latest_valid_fix = None
        self.transport_errors = []
        self.storage_errors = []
        self.storage_disabled = False
        self.pending_recorder = None
        self.pending_boundary = None

    def start(self):
        self.thread = threading.Thread(
            target=self._run,
            name="gps-nmea-reader",
            daemon=True,
        )
        self.thread.start()

    def rebind_recorder(self, recorder: "RunRecorder", boundary: float):
        """Ask the reader thread to reopen its log in a new run folder.

        The switch waits until a sentence's own receive time crosses the
        boundary, so a sentence received just before it is not filed under the
        next folder.  The switch happens in the reader thread between
        sentences, so no partially written line is split across two files.
        """
        with self.lock:
            self.pending_recorder = recorder
            self.pending_boundary = float(boundary)

    def _configure_serial(self, fd: int):
        import termios

        speed_name = f"B{self.baudrate}"
        speed = getattr(termios, speed_name, None)
        if speed is None:
            raise ValueError(f"unsupported GPS baudrate: {self.baudrate}")
        attrs = termios.tcgetattr(fd)
        attrs[0] = termios.IGNPAR
        attrs[1] = 0
        attrs[2] = termios.CS8 | termios.CREAD | termios.CLOCAL
        attrs[3] = 0
        attrs[4] = speed
        attrs[5] = speed
        attrs[6][termios.VMIN] = 0
        attrs[6][termios.VTIME] = 5
        termios.tcsetattr(fd, termios.TCSANOW, attrs)
        termios.tcflush(fd, termios.TCIFLUSH)

    def _open(self):
        candidates = discover_gps_serial_devices(
            self.device,
            self.preferred_device,
        )
        with self.lock:
            self.candidate_devices = list(candidates)
        if not candidates:
            raise FileNotFoundError(f"no GPS serial device matches {self.device!r}")
        if len(candidates) != 1:
            raise RuntimeError(
                "ambiguous GPS serial devices; connect exactly one or set "
                f"gps_device explicitly: {candidates}"
            )
        active_device = candidates[0]
        fd = os.open(
            active_device,
            os.O_RDONLY | os.O_NOCTTY | os.O_NONBLOCK,
        )
        try:
            self._configure_serial(fd)
        except Exception:
            os.close(fd)
            raise
        self.fd = fd
        with self.lock:
            self.active_device = active_device
        print(f"[GPS] serial opened: {active_device}", flush=True)

    def _close(self):
        fd, self.fd = self.fd, None
        with self.lock:
            self.active_device = None
        if fd is not None:
            try:
                os.close(fd)
            except OSError:
                pass

    def _storage_failed(self, handle, exc):
        with self.lock:
            self.storage_errors.append(f"{type(exc).__name__}: {exc}")
            self.storage_disabled = True
        print(f"[GPS WARN] GPS logging disabled after storage error: {exc}", flush=True)
        try:
            handle.close()
        except OSError:
            pass
        return None

    def _record(self, record: dict, handle):
        if handle is None or self.storage_disabled:
            return handle
        try:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            return handle
        except OSError as exc:
            return self._storage_failed(handle, exc)

    def _sync(self, handle):
        """Make the log durable, so a power cut loses at most GPS_SYNC_SEC of it."""
        if handle is None:
            return None
        try:
            handle.flush()
            os.fsync(handle.fileno())
            return handle
        except OSError as exc:
            return self._storage_failed(handle, exc)

    def _switch_log(self, handle, recorder):
        self.recorder = recorder
        if handle is not None:
            try:
                handle.flush()
                os.fsync(handle.fileno())
            except OSError:
                pass
            finally:
                try:
                    handle.close()
                except OSError:
                    pass
        try:
            new_handle = recorder.gps_jsonl.open("a", encoding="utf-8", buffering=1)
        except OSError as exc:
            with self.lock:
                self.storage_errors.append(f"{type(exc).__name__}: {exc}")
                self.storage_disabled = True
            print(f"[GPS WARN] cannot open GPS log after rotation: {exc}", flush=True)
            return None
        # A rotation gives a fresh file, so an earlier write failure on the old
        # one must not keep logging switched off.
        with self.lock:
            self.storage_disabled = False
        return new_handle

    def _consume_line(self, raw: bytes, handle):
        received_timestamp = time.time()
        with self.lock:
            pending = self.pending_recorder
            boundary = self.pending_boundary
        if pending is not None and received_timestamp >= boundary:
            with self.lock:
                self.pending_recorder = None
                self.pending_boundary = None
            handle = self._switch_log(handle, pending)
        text = raw.decode("ascii", errors="replace").strip()
        if not text:
            return handle
        record = parse_nmea_sentence(text, received_timestamp)
        record["monotonic_ns"] = time.monotonic_ns()
        with self.lock:
            self.sentence_count += 1
            self.last_sentence_timestamp = received_timestamp
            if not record.get("checksum_valid"):
                self.checksum_error_count += 1
            if record.get("parse_error"):
                self.parse_error_count += 1
            if record.get("valid"):
                self.valid_fix_count += 1
                self.last_valid_fix_timestamp = received_timestamp
                self.latest_valid_fix = dict(record)
        return self._record(record, handle)

    def _run(self):
        handle = None
        try:
            handle = self.recorder.gps_jsonl.open("a", encoding="utf-8", buffering=1)
        except OSError as exc:
            with self.lock:
                self.storage_errors.append(f"{type(exc).__name__}: {exc}")
                self.storage_disabled = True
            print(f"[GPS WARN] cannot open GPS log: {exc}", flush=True)

        buffer = bytearray()
        reconnect_attempt = 0
        next_sync = time.monotonic() + GPS_SYNC_SEC
        try:
            while not self.stop_event.is_set():
                if self.fd is None:
                    try:
                        self._open()
                        reconnect_attempt = 0
                        with self.lock:
                            self.receiver_open_count += 1
                    except (OSError, ValueError, RuntimeError) as exc:
                        reconnect_attempt += 1
                        delay = min(
                            GPS_RECONNECT_MAX_SEC,
                            GPS_RECONNECT_INITIAL_SEC * (2 ** min(reconnect_attempt - 1, 10)),
                        )
                        with self.lock:
                            self.transport_errors.append(f"{type(exc).__name__}: {exc}")
                            if len(self.transport_errors) > 20:
                                self.transport_errors = self.transport_errors[-20:]
                        self.stop_event.wait(delay)
                        continue

                try:
                    readable, _, _ = select.select([self.fd], [], [], 0.2)
                    if not readable:
                        # NMEA arrives in one burst per fix. Syncing in the quiet
                        # gap after a burst never delays a sentence's receive time.
                        if time.monotonic() >= next_sync:
                            handle = self._sync(handle)
                            next_sync = time.monotonic() + GPS_SYNC_SEC
                        continue
                    chunk = os.read(self.fd, 4096)
                    if not chunk:
                        raise OSError("GPS serial device returned EOF")
                    buffer.extend(chunk)
                    while b"\n" in buffer:
                        raw, _, remainder = buffer.partition(b"\n")
                        buffer = bytearray(remainder)
                        handle = self._consume_line(raw.rstrip(b"\x0d"), handle)
                    if len(buffer) > 16384:
                        buffer.clear()
                        with self.lock:
                            self.parse_error_count += 1
                except (OSError, ValueError) as exc:
                    with self.lock:
                        self.transport_errors.append(f"{type(exc).__name__}: {exc}")
                        if len(self.transport_errors) > 20:
                            self.transport_errors = self.transport_errors[-20:]
                    self._close()
                    buffer.clear()
                    handle = self._sync(handle)
        finally:
            self._close()
            if handle is not None:
                try:
                    handle.flush()
                    os.fsync(handle.fileno())
                except OSError as exc:
                    with self.lock:
                        self.storage_errors.append(f"{type(exc).__name__}: {exc}")
                finally:
                    handle.close()

    def is_connected(self) -> bool:
        """Serial device presence, independent of NMEA traffic or satellite fix."""
        with self.lock:
            device = self.active_device
            opened = self.fd is not None
        return bool(opened and device and os.path.exists(device))

    def stats(self) -> dict:
        with self.lock:
            latest_fix = None if self.latest_valid_fix is None else dict(self.latest_valid_fix)
            return {
                "enabled": True,
                "device": self.active_device or self.device,
                "device_setting": self.device,
                "preferred_device": self.preferred_device,
                "active_device": self.active_device,
                "candidate_devices": list(self.candidate_devices),
                "baudrate": self.baudrate,
                "sentence_count": self.sentence_count,
                "checksum_error_count": self.checksum_error_count,
                "parse_error_count": self.parse_error_count,
                "valid_fix_count": self.valid_fix_count,
                "last_sentence_timestamp": self.last_sentence_timestamp,
                "last_valid_fix_timestamp": self.last_valid_fix_timestamp,
                "latest_valid_fix": latest_fix,
                "receiver_open_count": self.receiver_open_count,
                "receiver_restart_count": max(0, self.receiver_open_count - 1),
                "transport_errors": list(self.transport_errors),
                "storage_errors": list(self.storage_errors),
                "storage_disabled": self.storage_disabled,
            }

    def stop(self):
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join(timeout=5.0)

def _finite_float(value) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None

def lidar__nearest_gps_record(
    stream: Optional[tuple[np.ndarray, list[dict]]], camera_timestamp: float
) -> tuple[Optional[dict], float]:
    if stream is None:
        return (None, float("inf"))
    times, records = stream
    if times.size == 0:
        return (None, float("inf"))
    pos = int(np.searchsorted(times, camera_timestamp))
    candidates: list[int] = []
    if pos < len(times):
        candidates.append(pos)
    if pos > 0:
        candidates.append(pos - 1)
    best_pos = min(candidates, key=lambda index: abs(float(times[index]) - camera_timestamp))
    error = abs(float(times[best_pos]) - camera_timestamp)
    if error > lidar__GPS_MAX_TIME_ERROR_SEC:
        return (None, error)
    return (records[best_pos], error)

def matched_gps_position(streams, timestamp):
    """Position of the nearest valid NMEA fix; latitude/longitude are None without one."""
    gga, _ = lidar__nearest_gps_record(streams.get("GGA"), timestamp)
    rmc, _ = lidar__nearest_gps_record(streams.get("RMC"), timestamp)
    def position_valid(record):
        return (record is not None and
                record.get("valid") is not False and record.get("checksum_valid") is not False and
                _finite_float(record.get("latitude_deg")) is not None and
                _finite_float(record.get("longitude_deg")) is not None and
                abs(float(record["latitude_deg"])) <= 90 and
                abs(float(record["longitude_deg"])) <= 180)
    valid_rmc = position_valid(rmc) and str(rmc.get("gps_status", rmc.get("status", ""))).upper() == "A"
    valid_gga = position_valid(gga) and (_finite_float(gga.get("fix_quality")) or 0) > 0
    position = rmc if valid_rmc else gga if valid_gga else None
    return dict(
        latitude_deg=None if position is None else position["latitude_deg"],
        longitude_deg=None if position is None else position["longitude_deg"],
    )

class GpsTail:
    """Read newly committed NMEA rows once, retaining only GGA/RMC."""

    def __init__(self, run):
        self.path = run / "gps/gps.jsonl"
        self.position = 0
        self.grouped = {"GGA": [], "RMC": []}
        self.identity = None

    def read(self):
        changed = False
        if not self.path.exists():
            return None
        stat = self.path.stat()
        identity = (stat.st_dev, stat.st_ino)
        if self.identity is not None and (
            identity != self.identity or stat.st_size < self.position
        ):
            raise RuntimeError(f"GPS manifest replaced or truncated: {self.path}")
        self.identity = identity
        with self.path.open("rb") as stream:
            stream.seek(self.position)
            while True:
                line = stream.readline()
                if not line.endswith(b"\n"):
                    break
                self.position = stream.tell()
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                    kind = str(row.get("sentence_type", "")).upper()
                    if kind in self.grouped and math.isfinite(float(row["timestamp"])):
                        self.grouped[kind].append(row)
                        changed = True
                except (ValueError, TypeError, KeyError):
                    continue
        if not changed:
            return None
        streams = {}
        for kind, rows in self.grouped.items():
            rows.sort(key=lambda row: float(row["timestamp"]))
            streams[kind] = (np.asarray([float(row["timestamp"]) for row in rows]), rows)
        return streams

def get_current_gps_fix(frame_timestamp, gps_streams=None):
    position = matched_gps_position(gps_streams or {}, frame_timestamp)
    return position["latitude_deg"], position["longitude_deg"]

class GpsSession:
    """Adapter for source GPS logs/history only; saver owns image/meta folders."""

    def __init__(self, now_kst, root_dir="/mnt/ssd/porthole_runs"):
        self.root_dir = root_dir
        self.session_str = now_kst.strftime("%Y%m%d_%H%M")
        gps_dir = os.path.join(root_dir, now_kst.strftime("%Y%m%d"), self.session_str, "gps")
        os.makedirs(gps_dir, exist_ok=True)
        self.gps_jsonl = Path(gps_dir) / "gps.jsonl"
        self.gps_tail = GpsTail(self.gps_jsonl.parent.parent)
        self.gps_streams = {}
        self.armed_session = None
        self.armed_boundary = None

def refresh_gps_session(session, gps_recorder, now_kst):
    """Retain source pre-boundary reservation and per-run history isolation."""
    session_str = now_kst.strftime("%Y%m%d_%H%M")
    if session.session_str != session_str:
        reserved = session.armed_session
        session = reserved if reserved is not None and reserved.session_str == session_str else GpsSession(now_kst, session.root_dir)
        if gps_recorder.recorder is not session and gps_recorder.pending_recorder is not session:
            gps_recorder.rebind_recorder(session, now_kst.replace(second=0, microsecond=0).timestamp())

    boundary = now_kst.replace(second=0, microsecond=0) + timedelta(minutes=1)
    if session.armed_session is None and (boundary - now_kst).total_seconds() <= 5.0:
        session.armed_session = GpsSession(boundary, session.root_dir)
        session.armed_boundary = boundary.timestamp()
        gps_recorder.rebind_recorder(session.armed_session, session.armed_boundary)

    streams = session.gps_tail.read()
    if streams is not None:
        session.gps_streams = streams
    return session

# --- [4] 딥엑스 PPU 추론 클래스 (공용) ---
class DeepXPPUModel:
    def __init__(self, engine_path, conf_thres=0.3):
        self.engine_path = engine_path
        self.conf_thres = conf_thres
        self.iou_thres = 0.45
        self.input_width = 640
        self.input_height = 640
        self.input_layout = "hwc"
        self.input_has_batch = False

        self.engine_pool = queue.Queue(maxsize=1)
        io = InferenceOption()
        engine = InferenceEngine(self.engine_path, io)
        self.engine_pool.put(engine)
        
        self._load_input_shape(engine)
        print(f"[AI] DeepX 모델 로드 완료: {engine_path}")

    def _load_input_shape(self, engine):
        try:
            input_info = engine.get_input_tensors_info()
            shape = list(input_info[0].get("shape", []))
        except Exception as e:
            return

        if len(shape) == 4:
            self.input_has_batch = True
            if shape[-1] in [1, 3, 4]:
                self.input_layout = "nhwc"
                self.input_height, self.input_width = int(shape[1]), int(shape[2])
            elif shape[1] in [1, 3, 4]:
                self.input_layout = "nchw"
                self.input_height, self.input_width = int(shape[2]), int(shape[3])
        elif len(shape) == 3:
            self.input_has_batch = False
            if shape[-1] in [1, 3, 4]:
                self.input_layout = "hwc"
                self.input_height, self.input_width = int(shape[0]), int(shape[1])
            elif shape[0] in [1, 3, 4]:
                self.input_layout = "chw"
                self.input_height, self.input_width = int(shape[1]), int(shape[2])

    def _prepare_input_tensor(self, npu_input):
        input_tensor = cv2.cvtColor(npu_input, cv2.COLOR_BGR2RGB)
        if self.input_layout in ["nchw", "chw"]:
            input_tensor = np.transpose(input_tensor, (2, 0, 1))
        if self.input_has_batch:
            input_tensor = np.expand_dims(input_tensor, axis=0)
        return np.ascontiguousarray(input_tensor, dtype=np.uint8)

    def infer(self, img, target_class_map):
        if img is None: return []

        h_orig, w_orig = img.shape[:2]
        
        # 640x480 입력을 NPU에 맞게 640x640 레터박스(상하 여백) 처리
        if w_orig != self.input_width or h_orig != self.input_height:
            scale = min(self.input_width / w_orig, self.input_height / h_orig)
            nw, nh = int(w_orig * scale), int(h_orig * scale)
            resized = cv2.resize(img, (nw, nh))
            canvas = np.full((self.input_height, self.input_width, 3), 114, dtype=np.uint8)
            dw, dh = (self.input_width - nw) // 2, (self.input_height - nh) // 2
            canvas[dh:dh+nh, dw:dw+nw] = resized
            npu_input = canvas
        else:
            npu_input = img
            scale = 1.0
            dw, dh = 0, 0

        input_tensor = self._prepare_input_tensor(npu_input)

        engine = self.engine_pool.get()
        try:
            output_tensor = engine.run([input_tensor])
            raw_dets = self.postprocess_ppu(output_tensor, self.conf_thres, self.iou_thres)
            
            res = []
            for box, score, cls_id in raw_dets:
                if cls_id not in target_class_map:
                    continue
                    
                # 레터박스 여백(dw, dh) 제거 및 원래 640x480 좌표로 복원
                x1 = np.clip((box[0] - dw) / scale, 0, w_orig)
                y1 = np.clip((box[1] - dh) / scale, 0, h_orig)
                x2 = np.clip((box[2] - dw) / scale, 0, w_orig)
                y2 = np.clip((box[3] - dh) / scale, 0, h_orig)
                
                eng_name, kor_name, cat_id = target_class_map[cls_id]
                res.append([int(x1), int(y1), int(x2), int(y2), float(score), eng_name, kor_name, int(cat_id)])
                
            return res
        except Exception as e:
            print(f"[AI Error] PPU 추론 실패: {e}")
            return []
        finally:
            self.engine_pool.put(engine)

    def postprocess_ppu(self, output_tensor, conf_thres, iou_thres):
        try:
            raw_data = output_tensor[0]
            if isinstance(raw_data, bytes):
                flat = np.frombuffer(raw_data, dtype=np.uint8).copy()
            else:
                flat = np.ascontiguousarray(raw_data).view(np.uint8).ravel().copy()

            if len(flat) == 0: return []
            stride = 32
            if len(flat) % stride != 0: return []

            flat_stride = flat.reshape(len(flat) // stride, stride)
            boxes_raw = np.ascontiguousarray(flat_stride[:, :16]).view(np.float32).reshape(-1, 4)
            scores = np.ascontiguousarray(flat_stride[:, 20:24]).view(np.float32).flatten()
            labels = np.ascontiguousarray(flat_stride[:, 24:28]).view(np.uint32).flatten()

            mask = scores >= conf_thres
            if not np.any(mask): return []

            boxes_raw = boxes_raw[mask]
            scores = scores[mask]
            labels = labels[mask]

            cx, cy = boxes_raw[:, 0], boxes_raw[:, 1]
            bw, bh = boxes_raw[:, 2], boxes_raw[:, 3]

            x1 = cx - bw * 0.5
            y1 = cy - bh * 0.5
            x2 = cx + bw * 0.5
            y2 = cy + bh * 0.5

            max_wh = 7680
            class_offset = labels.astype(np.float32) * max_wh
            boxes_shifted = np.column_stack([x1 + class_offset, y1 + class_offset, bw, bh])

            indices = cv2.dnn.NMSBoxes(boxes_shifted.tolist(), scores.tolist(), conf_thres, iou_thres)
            if indices is None or len(indices) == 0: return []

            results = []
            for i in np.array(indices).reshape(-1):
                results.append([[x1[i], y1[i], x2[i], y2[i]], scores[i], labels[i]])

            return results
        except Exception as e:
            return []

# --- [5] 데이터 저장 도우미 함수 ---
def save_detection_data(frame, detections, frame_id, terminal_id="terminal01", root_dir="/mnt/ssd/porthole_runs", *, frame_timestamp, gps_streams=None, gps_session=None, gps_recorder=None):
    now_kst = datetime.now()
    now_utc = datetime.fromtimestamp(frame_timestamp, timezone.utc)
    
    date_str = now_kst.strftime("%Y%m%d")
    session_str = now_kst.strftime("%Y%m%d_%H%M")
    
    base_dir = os.path.join(root_dir, date_str, session_str)
    frames_dir = os.path.join(base_dir, "frames")
    gps_dir = os.path.join(base_dir, "gps")
    meta_dir = os.path.join(base_dir, "meta")
    
    for d in [frames_dir, gps_dir, meta_dir]:
        os.makedirs(d, exist_ok=True)
        
    timestamp_ns = time.time_ns()
    file_base = f"frame_{timestamp_ns}_{frame_id:08d}"
    
    jpg_filename = f"{file_base}.jpg"
    json_filename = f"{file_base}.json"
    
    if gps_session is not None:
        gps_session = refresh_gps_session(gps_session, gps_recorder, now_kst)
        gps_streams = gps_session.gps_streams
    lat, lon = get_current_gps_fix(frame_timestamp, gps_streams)
    record_id = f"{terminal_id}/{session_str}/{frame_id}"
    
    categories_dict = {}
    annotations = []
    
    for i, (x1, y1, x2, y2, score, eng_name, kor_name, cat_id) in enumerate(detections):
        categories_dict[cat_id] = eng_name
        w = x2 - x1
        h = y2 - y1
        annotations.append({
            "id": i + 1,
            "image_id": 1,
            "category_id": cat_id, 
            "confidence": round(float(score), 4),
            "bbox": [int(x1), int(y1), int(w), int(h)],
            "segmentation": [[int(x1), int(y1), int(x2), int(y1), int(x2), int(y2), int(x1), int(y2)]],
            "measurements": {
                "size": {"length_m": None, "width_m": None, "area_m2": None}, 
                "depth": {"median_cm": None}
            }
        })
        
    categories = [{"id": cid, "name": name} for cid, name in sorted(categories_dict.items())]
    
    images = [{
        "id": 1,
        "width": frame.shape[1],
        "height": frame.shape[0],
        "file_name": jpg_filename,
        "date_captured": now_utc.strftime("%Y-%m-%dT%H:%M:%SZ") 
    }]
        
    data = {
        "record_id": record_id,
        "categories": categories,
        "images": images,
        "annotations": annotations,
        "gps": {"latitude_deg": lat, "longitude_deg": lon}, 
        "lidar": {"pcap_files": []}
    }
    
    cv2.imwrite(os.path.join(frames_dir, jpg_filename), frame)
    with open(os.path.join(meta_dir, json_filename), 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return gps_session

# --- [6] FFmpeg 듀얼 스트림 읽기 ---
def read_exact(pipe, size):
    data = bytearray(size)
    view = memoryview(data)
    bytes_read = 0
    while bytes_read < size:
        chunk = pipe.stdout.read(size - bytes_read)
        if not chunk: return None
        view[bytes_read:bytes_read+len(chunk)] = chunk
        bytes_read += len(chunk)
    return bytes(data)

class FFmpegStreamReader(threading.Thread):
    def __init__(self, url, width, height, buffer, is_sub=False):
        super().__init__(daemon=True)
        self.url = url
        self.w = width
        self.h = height
        self.buffer = buffer
        self.is_sub = is_sub
        self.running = True

    def run(self):
        command = [
            'ffmpeg', '-nostdin', '-hwaccel', 'drm',
            '-rtsp_transport', 'tcp', '-fflags', 'nobuffer',
            '-flags', 'low_delay', '-i', self.url,
            '-f', 'image2pipe', '-pix_fmt', 'nv12',
            '-vcodec', 'rawvideo', '-'
        ]
        frame_size = int(self.w * self.h * 1.5)
        pipe = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=10**8)
        
        frame_id = 0
        try:
            while self.running:
                raw_data = read_exact(pipe, frame_size)
                if not raw_data: break
                frame_timestamp = time.time() if not self.is_sub else None
                
                frame_id += 1
                if self.is_sub and frame_id % 15 != 0:
                    continue

                yuv = np.frombuffer(raw_data, dtype='uint8').reshape((int(self.h * 1.5), self.w))
                bgr = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_NV12)
                self.buffer.put(bgr.copy() if self.is_sub else (bgr.copy(), frame_timestamp))
        finally:
            pipe.terminate()

# --- [7] AI 백그라운드 워커 스레드 ---
def ai_worker_loop(pothole_det, roadobj_det, sub_frame_buffer, result_buffer, stop_event, scale_x, scale_y):
    """Sub(640x480) 프레임에서 추론 후 Main(1080P) 해상도로 매핑합니다."""
    while not stop_event.is_set():
        frame = sub_frame_buffer.get_and_clear()
        if frame is None:
            time.sleep(0.005)
            continue
            
        # 모델별 독립 추론 (각자의 Target Map 적용)
        dets_pothole = pothole_det.infer(frame, POTHOLE_CLASSES)
        dets_roadobj = roadobj_det.infer(frame, ROADOBJ_CLASSES)
        
        combined_raw = dets_pothole + dets_roadobj
        scaled_detections = []
        for x1, y1, x2, y2, score, eng_name, kor_name, cat_id in combined_raw:
            sx1, sy1 = int(x1 * scale_x), int(y1 * scale_y)
            sx2, sy2 = int(x2 * scale_x), int(y2 * scale_y)
            scaled_detections.append([sx1, sy1, sx2, sy2, score, eng_name, kor_name, cat_id])
            
        result_buffer.put(scaled_detections)

# --- [8] 메인 파이프라인 ---
def main():
    W_MAIN, H_MAIN = 1920, 1080
    main_url = "rtsp://admin:Hu924688@192.168.11.64:554/Streaming/Channels/101"
    
    # 서브스트림 해상도 640x480 적용
    W_SUB, H_SUB = 640, 480  
    sub_url = "rtsp://admin:Hu924688@192.168.11.64:554/Streaming/Channels/102"
    
    main_buffer = LatestItemBuffer()
    sub_buffer = LatestItemBuffer()
    result_buffer = LatestItemBuffer()
    stop_event = threading.Event()

    main_reader = FFmpegStreamReader(main_url, W_MAIN, H_MAIN, main_buffer, is_sub=False)
    main_reader.start()
    
    sub_reader = FFmpegStreamReader(sub_url, W_SUB, H_SUB, sub_buffer, is_sub=True)
    sub_reader.start()

    scale_x = W_MAIN / W_SUB
    scale_y = H_MAIN / H_SUB
    
    # 2개 모델 초기화
    ai_pothole = DeepXPPUModel(engine_path="pothole_best_ppu.dxnn", conf_thres=0.35)
    ai_roadobj = DeepXPPUModel(engine_path="roadobj_ppu.dxnn", conf_thres=0.35)

    ai_thread = threading.Thread(
        target=ai_worker_loop, 
        args=(ai_pothole, ai_roadobj, sub_buffer, result_buffer, stop_event, scale_x, scale_y), 
        daemon=True
    )
    ai_thread.start()

    print("듀얼 모델 & 듀얼 스트림(Main: 1080P, Sub: 480P) 모니터링 시작...")
    
    main_frame_count = 0
    last_saved_detections = None
    gps_recorder = None
    
    try:
        gps_session = GpsSession(datetime.now())
        gps_recorder = GpsNmeaRecorder(GPS_DEVICE, GPS_BAUDRATE, gps_session, GPS_PREFERRED_DEVICE)
        gps_session = refresh_gps_session(gps_session, gps_recorder, datetime.now())
        gps_recorder.start()
        while True:
            gps_session = refresh_gps_session(gps_session, gps_recorder, datetime.now())
            frame_item = main_buffer.get_and_clear()
            if frame_item is None:
                time.sleep(0.005)
                continue
            frame_main, frame_timestamp = frame_item
                
            main_frame_count += 1
            current_detections = result_buffer.get_current() or []
            
            if current_detections and current_detections != last_saved_detections:
                gps_session = save_detection_data(frame_main.copy(), current_detections, main_frame_count, frame_timestamp=frame_timestamp, gps_session=gps_session, gps_recorder=gps_recorder)
                last_saved_detections = current_detections

            for x1, y1, x2, y2, score, eng_name, kor_name, cat_id in current_detections:
                display_text = f"{eng_name} ({score:.2f})"
                cv2.rectangle(frame_main, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame_main, display_text, (x1, max(20, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            cv2.imshow('101 Main Stream (Dual Model Overlay)', frame_main)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except Exception as e:
        print(f"오류 발생: {e}")

    finally:
        stop_event.set()
        main_reader.running = False
        sub_reader.running = False
        try:
            if gps_recorder is not None:
                if gps_recorder.thread is None or gps_recorder.thread.ident is not None:
                    gps_recorder.stop()
                else:
                    gps_recorder.stop_event.set()  # Thread.start failed; joining is invalid.
        finally:
            cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
