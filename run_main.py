import os
import cv2
import numpy as np
import subprocess
import threading
import time
import json
import signal
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


# --- [1-2] 화면 출력 설정 ---
# HEADLESS=1 이면 화면 없이 실행, HEADLESS=0 이면 강제 표시.
# 지정하지 않으면 DISPLAY/WAYLAND_DISPLAY가 없을 때(systemd 서비스 등) 자동으로 화면 없이 실행.
_headless_env = os.environ.get("HEADLESS", "").strip().lower()
if _headless_env in ("1", "true", "yes"):
    SHOW_WINDOW = False
elif _headless_env in ("0", "false", "no"):
    SHOW_WINDOW = True
else:
    SHOW_WINDOW = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))

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
        # Completion adapter must not observe the new owner before the old log closes.
        with self.lock:
            self.recorder = recorder
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
                        print(f"[GPS WARN] open failed (retry {reconnect_attempt}, {delay:.0f}s 후): {exc}", flush=True)
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
                    print(f"[GPS WARN] serial read error, reconnecting: {exc}", flush=True)
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
            # 파일이 교체/잘림 → 프로그램을 멈추지 않고 처음부터 다시 읽음
            print(f"[GPS WARN] GPS log replaced/truncated, re-reading: {self.path}", flush=True)
            self.position = 0
            self.grouped = {"GGA": [], "RMC": []}
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

    def __init__(self, now_kst, root_dir="/media/hucomputer/DISK/pothole_runs"):
        self.root_dir = root_dir
        now_kst = now_kst.replace(minute=now_kst.minute // 10 * 10, second=0, microsecond=0)
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
    bucket = now_kst.replace(minute=now_kst.minute // 10 * 10, second=0, microsecond=0)
    session_str = bucket.strftime("%Y%m%d_%H%M")
    if session.session_str != session_str:
        reserved = session.armed_session
        session = reserved if reserved is not None and reserved.session_str == session_str else GpsSession(now_kst, session.root_dir)
        if gps_recorder.recorder is not session and gps_recorder.pending_recorder is not session:
            gps_recorder.rebind_recorder(session, bucket.timestamp())

    boundary = bucket + timedelta(minutes=10)
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
    def __init__(self, engine_path, conf_thres=0.2):
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

def save_detection_data(frame, detections, frame_id, terminal_id="axsprint-rasp-01", root_dir="/media/hucomputer/DISK/pothole_runs", *, frame_timestamp, gps_streams=None, gps_session=None, gps_recorder=None, session_uploads=None):
    now_kst = datetime.now()
    now_utc = datetime.fromtimestamp(frame_timestamp, timezone.utc)

    date_str = now_kst.strftime("%Y%m%d")
    session_str = now_kst.replace(minute=now_kst.minute // 10 * 10, second=0, microsecond=0).strftime("%Y%m%d_%H%M")

    base_dir = os.path.join(root_dir, date_str, session_str)
    if session_uploads is not None:
        session_uploads.track(base_dir)
    frames_dir = os.path.join(base_dir, "frames")
    frames_bbox_dir = os.path.join(base_dir, "frames_bbox")
    gps_dir = os.path.join(base_dir, "gps")
    meta_dir = os.path.join(base_dir, "meta")
    for d in [frames_dir, frames_bbox_dir, gps_dir, meta_dir]:
        os.makedirs(d, exist_ok=True)

    # 명세: 파일명 시각 = 촬영 시각(date_captured)
    timestamp_ns = int(frame_timestamp * 1e9)
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

    jpg_path = os.path.join(frames_dir, jpg_filename)
    if not cv2.imwrite(jpg_path, frame):
        print(f"[SAVE WARN] 이미지 저장 실패: {jpg_path}", flush=True)
        return gps_session

    bbox_path = os.path.join(frames_bbox_dir, jpg_filename)
    try:
        bbox_frame = frame.copy()
        for x1, y1, x2, y2, score, eng_name, kor_name, cat_id in detections:
            display_text = f"{eng_name} ({score:.2f})"
            cv2.rectangle(bbox_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(bbox_frame, display_text, (x1, max(20, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        if not cv2.imwrite(bbox_path, bbox_frame):
            print(f"[SAVE WARN] bbox 이미지 저장 실패: {bbox_path}", flush=True)
    except Exception as e:
        print(f"[SAVE WARN] bbox 이미지 저장 실패: {bbox_path} ({type(e).__name__})", flush=True)


    data = {
        "record_id": record_id,
        "categories": categories,
        "images": images,
        "annotations": annotations,
        "gps": {"latitude_deg": lat, "longitude_deg": lon},
        "lidar": {"pcap_files": []}
    }

    json_path = os.path.join(meta_dir, json_filename)
    tmp_path = json_path + ".tmp"
    with open(tmp_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, json_path)  # 중간에 끊겨도 반쯤 쓰인 JSON이 남지 않음
    if session_uploads is not None:
        stage_saved_frame(session_uploads.detections, Path(base_dir), file_base, frame_id)
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
def append_detection_log(detections, root_dir="/media/hucomputer/DISK/pothole_runs", *, session_uploads=None):
    """Append one row per filtered object, independently of image saving."""
    if not detections:
        return
    detected_at = datetime.now(timezone.utc)
    now_local = detected_at.astimezone()
    now_local = now_local.replace(minute=now_local.minute // 10 * 10, second=0, microsecond=0)
    logs_dir = os.path.join(root_dir, now_local.strftime("%Y%m%d"),
                            now_local.strftime("%Y%m%d_%H%M"), "logs")
    lines = [json.dumps({"class_name": det[5], "conf": float(det[4]),
                         "detected_at": detected_at.isoformat()},
                        ensure_ascii=False, allow_nan=False) + "\n"
             for det in detections]
    if session_uploads is not None:
        session_uploads.track(os.path.dirname(logs_dir))
    os.makedirs(logs_dir, exist_ok=True)
    with open(os.path.join(logs_dir, "detections.jsonl"), "a", encoding="utf-8") as f:
        f.writelines(lines)
        f.flush()



def ai_worker_loop(pothole_det, roadobj_det, sub_frame_buffer, result_buffer, stop_event, scale_x, scale_y, session_uploads=None):
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

        try:
            if session_uploads is None:
                append_detection_log(scaled_detections)
            else:
                session_uploads.log(scaled_detections)
        except Exception as e:
            print(f"[DETECTION LOG WARN] {type(e).__name__}; continuing", flush=True)
        result_buffer.put(scaled_detections)

# --- [8] 메인 파이프라인 ---
# Transfer contract copied from pothole/live_detection@302d7182fdee360a0b733016b88643f2e8026aa9.
# Keep the original HTTP and SSH wire protocols and source queue/drop/retry policies.
import hashlib
import re
import shlex
import shutil
import socket
import sys
import tempfile
import urllib.error
import urllib.request
import uuid
import zlib
from dataclasses import dataclass
from types import SimpleNamespace

PROJECT = Path(__file__).resolve().parent
UPLOAD_STAGING = Path(tempfile.gettempdir()) / "pothole_rasp_upload"
RAW_DONE = PROJECT / "var" / "raw_uploaded.txt"
RAW_LOCK = Path(tempfile.gettempdir()) / "pothole_rasp_raw_upload.lock"


DETECTIONS_CSV = "porthole_detections.csv"


REMOTE_FOLDER = "porthole_live_analysis"  # server: <PORTHOLE_UPLOAD_DIR>/<this>/<run>/certifcate


UPLOAD_QUEUE_LIMIT = 100  # frames waiting to be sent; more are only listed in the CSV


OFFLINE_RETRY_SEC = 30  # after a failed send, new detections are only listed in the CSV


def server_transport_source():
    """Build the stdlib-only receiver from the functions implemented in this file."""
    import inspect

    imports = "import hashlib, json, os, shutil, sys, time, uuid, zlib\nfrom pathlib import Path\n"
    constants = f"CHUNK = {CHUNK!r}\nMAX_HEADER = {MAX_HEADER!r}\nMAX_FRAME = {MAX_FRAME!r}\nPROTOCOL = {PROTOCOL!r}\n"
    source = "\n".join(
        inspect.getsource(f)
        for f in [
            artifact_pair_error,
            validate_manifest,
            clock_stable,
            content_aliases,
            fsync_dir,
            decode_wire_payload,
            receive_frame_payload,
            persist_frame_payload,
            serve,
        ]
    )
    return imports + constants + source + "\nserve(sys.argv[2])\n"


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


CHUNK = 64 * 1024


MAX_HEADER = 4 * 1024 * 1024


MAX_FRAME = 256 * 1024 * 1024


PROTOCOL = "porthole_artifacts_v4"


def artifact_pair_error(files):
    """Only one same-stem JPEG/JSON pair may reach the receiver."""
    if not isinstance(files, dict) or len(files) != 2:
        return "Upload requires exactly one JPG and one JSON"
    if any(not isinstance(name, str) for name in files):
        return "Invalid artifact filename"
    if {Path(name).suffix for name in files} != {".jpg", ".json"}:
        return "Upload only accepts JPG and JSON"
    if len({Path(name).stem for name in files}) != 1:
        return "JPG and JSON must have the same basename"
    return ""


def validate_manifest(manifest):
    token = manifest.get("_receive_token", "")
    if len(token) != 32 or any((c not in "0123456789abcdef" for c in token)):
        raise ValueError("Invalid receipt token")
    if type(manifest.get("frame_index")) is not int:
        raise ValueError("Invalid frame index")
    files = manifest.get("files")
    if not isinstance(files, dict) or not files or len(files) > 4096:
        raise ValueError("Invalid artifact list")
    error = artifact_pair_error(files)
    if error:
        raise ValueError(error)
    total = 0
    for name, info in files.items():
        if (
            not isinstance(name, str)
            or not name
            or name in (".", "..")
            or ("/" in name)
            or ("\\" in name)
            or ("\x00" in name)
            or (Path(name).name != name)
            or Path(name).suffix not in (".jpg", ".json")
        ):
            raise ValueError("Invalid artifact filename")
        size = info.get("bytes")
        digest = info.get("sha256", "")
        if (
            type(size) is not int
            or size < 0
            or len(digest) != 64
            or any((c not in "0123456789abcdef" for c in digest))
        ):
            raise ValueError("Invalid artifact size/hash")
        total += size
    if total > MAX_FRAME:
        raise ValueError("Artifact frame exceeds transport limit")
    return total


def clock_stable(wall, mono):
    return abs(time.time_ns() - wall - (time.monotonic_ns() - mono)) < 10000000


def content_aliases(files):
    """Send identical image bytes once, retaining every final artifact file."""
    seen, aliases = ({}, {})
    for name, info in files.items():
        identity = (info["bytes"], info["sha256"])
        if identity in seen:
            aliases[name] = seen[identity]
        else:
            seen[identity] = name
    return aliases


def fsync_dir(path):
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def payload_chunks(source, files, aliases):
    """Read and verify the exact bytes before transmission, coalescing small files."""
    pending = bytearray()
    for name, info in files.items():
        digest = hashlib.sha256()
        with (source / name).open("rb") as stream:
            if os.fstat(stream.fileno()).st_size != info["bytes"]:
                raise ValueError("Source artifact size changed")
            remaining = info["bytes"]
            while remaining:
                block = stream.read(min(CHUNK - len(pending), remaining))
                if not block:
                    raise EOFError("Source artifact truncated")
                digest.update(block)
                remaining -= len(block)
                if name not in aliases:
                    pending.extend(block)
                    if len(pending) == CHUNK:
                        yield bytes(pending)
                        pending.clear()
            if stream.read(1) or digest.hexdigest() != info["sha256"]:
                raise ValueError("Source artifact checksum changed: " + name)
    if pending:
        yield bytes(pending)


@dataclass(frozen=True)
class PreparedUpload:
    manifest: dict
    aliases: dict
    chunks: tuple
    payload_bytes: int
    wire_bytes: int
    wire_encoding: str
    wire_segments: tuple
    preparation_ms: float


def prepare_upload(source, manifest):
    """Complete source I/O, hashes and optional lossless encoding before send time."""
    begin = time.monotonic_ns()
    manifest = dict(manifest, _receive_token=uuid.uuid4().hex)
    manifest["files"] = {name: dict(info) for name, info in manifest["files"].items()}
    validate_manifest(manifest)
    aliases = content_aliases(manifest["files"])
    expected = sum(info["bytes"] for name, info in manifest["files"].items() if name not in aliases)
    encoded_files, segments = [], []
    for name, info in manifest["files"].items():
        raw = b"".join(payload_chunks(Path(source), {name: info}, aliases))
        if name in aliases:
            continue  # Alias content was verified but is sent only once.
        encoded, encoding = raw, "identity"
        # JPEG already has compression. Recompressing it wasted most CPU time.
        if Path(name).suffix.lower() == ".json" and len(raw) >= 1024:
            compressed = zlib.compress(raw, 1)
            if len(compressed) <= len(raw) - max(256, int(len(raw) * 0.05)):
                encoded, encoding = compressed, "zlib"
        encoded_files.append(encoded)
        segments.append(dict(name=name, wire_encoding=encoding, wire_bytes=len(encoded)))
    wire = b"".join(encoded_files)
    encoding = "files" if any(s["wire_encoding"] == "zlib" for s in segments) else "identity"
    return PreparedUpload(
        manifest,
        aliases,
        (wire,),
        expected,
        len(wire),
        encoding,
        tuple(segments),
        (time.monotonic_ns() - begin) / 1e6,
    )


def decode_wire_payload(wire, encoding, payload_bytes):
    if encoding == "identity":
        if len(wire) != payload_bytes:
            raise ValueError("Uncompressed payload size mismatch")
        return wire
    if encoding != "zlib" or not 0 < len(wire) < payload_bytes:
        raise ValueError("Compressed payload size mismatch")
    decoder = zlib.decompressobj()
    payload = decoder.decompress(wire, payload_bytes + 1)
    if (
        len(payload) != payload_bytes
        or not decoder.eof
        or decoder.unconsumed_tail
        or decoder.unused_data
    ):
        raise ValueError("Invalid or oversized decoded payload")
    return payload


def receive_frame_payload(stream, request, payload_bytes):
    """Bounded receive/decode. Receipt means all ORIGINAL payload bytes are ready."""
    encoding = request.get("wire_encoding")
    wire_bytes = request.get("wire_bytes")
    if (
        encoding not in ("identity", "zlib", "files")
        or type(wire_bytes) is not int
        or not 0 <= wire_bytes <= MAX_FRAME
    ):
        raise ValueError("Invalid payload encoding/size")
    if encoding == "identity" and wire_bytes != payload_bytes:
        raise ValueError("Uncompressed payload size mismatch")
    if encoding == "zlib" and not 0 < wire_bytes < payload_bytes:
        raise ValueError("Compressed payload size mismatch")
    segments = []
    if encoding == "files":
        aliases = content_aliases(request["files"])
        names = [name for name in request["files"] if name not in aliases]
        segments = request.get("wire_segments")
        if (
            not isinstance(segments, list)
            or len(segments) != len(names)
            or any(not isinstance(s, dict) for s in segments)
            or [s.get("name") for s in segments] != names
        ):
            raise ValueError("Invalid file payload mapping")
        for segment in segments:
            size = segment.get("wire_bytes")
            kind = segment.get("wire_encoding")
            raw_size = request["files"][segment["name"]]["bytes"]
            if (
                type(size) is not int
                or kind not in ("identity", "zlib")
                or (kind == "identity" and size != raw_size)
                or (kind == "zlib" and not 0 < size < raw_size)
            ):
                raise ValueError("Invalid file payload encoding/size")
        if sum(s["wire_bytes"] for s in segments) != wire_bytes:
            raise ValueError("File payload sizes do not match wire bytes")
    wall, mono = time.time_ns(), time.monotonic_ns()
    wire = stream.read(wire_bytes)
    wire_received_ns = time.monotonic_ns()
    if len(wire) != wire_bytes:
        raise EOFError("Artifact payload disconnected")
    if encoding == "files":
        offset, restored = 0, []
        for segment in segments:
            end = offset + segment["wire_bytes"]
            restored.append(
                decode_wire_payload(
                    wire[offset:end],
                    segment["wire_encoding"],
                    request["files"][segment["name"]]["bytes"],
                )
            )
            offset = end
        payload = b"".join(restored)
        if len(payload) != payload_bytes:
            raise ValueError("Restored payload size mismatch")
    else:
        payload = decode_wire_payload(wire, encoding, payload_bytes)
    # Keep decode time inside receive_gap: do not report compressed bytes as a
    # complete original JPG/JSON payload before they have been restored.
    received_wall, received_mono = time.time_ns(), time.monotonic_ns()
    timing = dict(
        server_received_epoch_ns=received_wall,
        server_received_monotonic_ns=received_mono,
        server_receiver_clock_stable=clock_stable(wall, mono),
        server_payload_receive_ms=(received_mono - mono) / 1e6,
        server_wire_receive_ms=(wire_received_ns - mono) / 1e6,
        server_payload_decode_ms=(received_mono - wire_received_ns) / 1e6,
        wire_bytes=wire_bytes,
        wire_encoding=encoding,
    )
    return payload, timing, wall, mono


def persist_frame_payload(target, request, aliases, payload):
    """Verify and persist original files after complete in-memory receipt."""
    stage = target / (".pending_upload_" + request["_receive_token"])
    stage.mkdir(mode=448)
    staged, payload_offset, write_ns = [], 0, 0
    payload_view = memoryview(payload)
    try:
        for name, info in request["files"].items():
            path = stage / name
            staged.append(path)
            if name in aliases:
                started = time.monotonic_ns()
                shutil.copyfile(stage / aliases[name], path)
                write_ns += time.monotonic_ns() - started
                continue
            remaining = info["bytes"]
            with path.open("xb", buffering=0) as output:
                while remaining:
                    size = min(CHUNK, remaining)
                    view = payload_view[payload_offset : payload_offset + size]
                    payload_offset += size
                    started = time.monotonic_ns()
                    while view:
                        written = output.write(view)
                        if not written:
                            raise OSError("Artifact write failed")
                        view = view[written:]
                    write_ns += time.monotonic_ns() - started
                    remaining -= size
        for name, info in request["files"].items():
            digest = hashlib.sha256()
            with (stage / name).open("rb") as source:
                for block in iter(lambda: source.read(CHUNK), b""):
                    digest.update(block)
                if (
                    os.fstat(source.fileno()).st_size != info["bytes"]
                    or digest.hexdigest() != info["sha256"]
                ):
                    raise ValueError("Server artifact checksum mismatch: " + name)
                os.fsync(source.fileno())
        for name in request["files"]:
            os.replace(stage / name, target / name)
        fsync_dir(target)
        return write_ns / 1e6
    finally:
        for path in staged:
            path.unlink(missing_ok=True)
        stage.rmdir()


def serve(target):
    target = Path(target)
    target.mkdir(parents=True, exist_ok=True)
    stream = sys.stdin.buffer

    def reply(payload):
        print(json.dumps(payload, separators=(",", ":")), flush=True)

    reply(dict(ready=PROTOCOL))
    while True:
        line = stream.readline(MAX_HEADER + 1)
        if not line:
            return
        if len(line) > MAX_HEADER or not line.endswith(b"\n"):
            raise ValueError("Invalid protocol header")
        request = json.loads(line)
        if request.get("op") == "probe":
            t2 = time.time_ns()
            reply(dict(t2=t2, m2=time.monotonic_ns(), t3=time.time_ns()))
            continue
        if request.get("op") != "frame":
            raise ValueError("Unknown operation")
        total = validate_manifest(request)
        aliases = content_aliases(request["files"])
        if request.get("aliases", {}) != aliases:
            raise ValueError("Invalid duplicate-content mapping")
        payload_bytes = sum(
            info["bytes"] for name, info in request["files"].items() if name not in aliases
        )
        payload, timing, wall, mono = receive_frame_payload(stream, request, payload_bytes)
        write_ms = persist_frame_payload(target, request, aliases, payload)
        reply(
            dict(
                token=request["_receive_token"],
                frame_index=request["frame_index"],
                files=len(request["files"]),
                bytes=total,
                payload_bytes=payload_bytes,
                **timing,
                server_receipt_clock_stable=clock_stable(
                    timing["server_received_epoch_ns"],
                    timing["server_received_monotonic_ns"],
                ),
                server_clock_stable=clock_stable(wall, mono),
                server_verified_epoch_ns=time.time_ns(),
                hashes_verified=True,
                definition="complete_payload_in_memory_before_file_write_v3",
                transport=PROTOCOL,
                server_payload_write_ms=write_ms,
            )
        )


class PersistentUploader:
    def __init__(self, options, relative, timeout=60):
        self.options, self.relative, self.timeout = (options, relative, timeout)
        self.proc = None
        self.buffer = bytearray()

    def _write(self, payload, deadline):
        view = memoryview(payload)
        fd = self.proc.stdin.fileno()
        while view:
            remaining = deadline - time.monotonic()
            if remaining <= 0 or not select.select([], [fd], [], remaining)[1]:
                raise TimeoutError("SSH upload write timed out")
            try:
                n = os.write(fd, view[:CHUNK])
            except BlockingIOError:
                continue
            if not n:
                raise ConnectionError("SSH upload disconnected")
            view = view[n:]

    def _json(self, payload, deadline):
        encoded = json.dumps(payload, separators=(",", ":")).encode() + b"\n"
        if len(encoded) > MAX_HEADER:
            raise ValueError("Upload header too large")
        self._write(encoded, deadline)

    def _read(self, deadline):
        fd = self.proc.stdout.fileno()
        while b"\n" not in self.buffer:
            remaining = deadline - time.monotonic()
            if remaining <= 0 or not select.select([fd], [], [], remaining)[0]:
                raise TimeoutError("SSH upload response timed out")
            block = os.read(fd, CHUNK)
            if not block:
                raise ConnectionError("SSH upload disconnected before receipt")
            self.buffer.extend(block)
            if len(self.buffer) > MAX_HEADER:
                raise ValueError("Upload response too large")
        line, _, remainder = self.buffer.partition(b"\n")
        self.buffer = bytearray(remainder)
        return json.loads(line)

    def connect(self):
        if self.proc is not None and self.proc.poll() is None:
            return
        self.close()
        target = self.options.destination.rstrip("/") + "/" + self.relative.as_posix()
        command = [
            "ssh",
            "-T",
            "-i",
            str(self.options.key),
            "-o",
            "BatchMode=yes",
            "-o",
            "StrictHostKeyChecking=yes",
            "-o",
            "ConnectTimeout=10",
            "-o",
            "ServerAliveInterval=15",
            "-o",
            "ServerAliveCountMax=2",
            f"{self.options.user}@{self.options.host}",
            "python3 -u -c "
            + shlex.quote(SERVER_TRANSPORT_SOURCE)
            + " --serve "
            + shlex.quote(target),
        ]
        self.proc = subprocess.Popen(
            command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, bufsize=0
        )
        os.set_blocking(self.proc.stdin.fileno(), False)
        self.buffer.clear()
        try:
            if self._read(time.monotonic() + self.timeout) != {"ready": PROTOCOL}:
                raise RuntimeError("SSH upload handshake failed")
        except Exception:
            self.close()
            raise

    def _send_payload(self, prepared, deadline):
        rate = self.options.bw_kib * 1024
        wall, mono = time.time_ns(), time.monotonic_ns()
        started, sent, wait_ns = mono / 1e9, 0, 0
        for chunk in prepared.chunks:
            for offset in range(0, len(chunk), CHUNK):
                block = memoryview(chunk)[offset : offset + CHUNK]
                delay = started + max(0, sent + len(block) - CHUNK) / rate - time.monotonic()
                if delay > 0:
                    if time.monotonic() + delay >= deadline:
                        raise TimeoutError("Upload bandwidth deadline exceeded")
                    before = time.monotonic_ns()
                    time.sleep(delay)
                    wait_ns += time.monotonic_ns() - before
                self._write(block, deadline)
                sent += len(block)
        if sent != prepared.wire_bytes:
            raise ValueError("Sent payload size mismatch")
        return dict(
            client_payload_started_epoch_ns=wall,
            client_payload_started_monotonic_ns=mono,
            client_payload_send_ms=(time.monotonic_ns() - mono) / 1e6,
            client_bandwidth_wait_ms=wait_ns / 1e6,
            client_payload_prepare_ms=prepared.preparation_ms,
        )

    def _clock_receipt(self, result, deadline):
        samples, probes = [], []
        for _ in range(5):
            t1, m1 = time.time_ns(), time.monotonic_ns()
            self._json({"op": "probe"}, deadline)
            probes.append((t1, m1))
        for t1, m1 in probes:
            probe = self._read(deadline)
            t4, m4 = time.time_ns(), time.monotonic_ns()
            if (
                abs(
                    probe["t2"]
                    - result["server_received_epoch_ns"]
                    - (probe["m2"] - result["server_received_monotonic_ns"])
                )
                >= 10000000
            ):
                result["server_receipt_clock_stable"] = False
            if abs(t4 - t1 - (m4 - m1)) < 10000000:
                rtt = t4 - t1 - (probe["t3"] - probe["t2"])
                if rtt >= 0:
                    samples.append((rtt, (probe["t2"] - t1 + probe["t3"] - t4) / 2))
        result.update(
            client_ack_epoch_ns=time.time_ns(),
            client_ack_monotonic_ns=time.monotonic_ns(),
        )
        if samples and result.get("server_clock_stable"):
            rtt, offset = min(samples)
            result.update(clock_offset_ns=offset, clock_uncertainty_ms=rtt / 2000000.0)

    def upload(self, source, manifest):
        try:
            prepared = prepare_upload(source, manifest)
            self.connect()
            deadline = time.monotonic() + max(
                self.timeout,
                prepared.wire_bytes / (self.options.bw_kib * 1024) * 2 + 10,
            )
            request = prepared.manifest
            self._json(
                dict(
                    op="frame",
                    frame_index=request["frame_index"],
                    _receive_token=request["_receive_token"],
                    files=request["files"],
                    aliases=prepared.aliases,
                    wire_bytes=prepared.wire_bytes,
                    wire_encoding=prepared.wire_encoding,
                    wire_segments=prepared.wire_segments,
                ),
                deadline,
            )
            send_timing = self._send_payload(prepared, deadline)
            result = self._read(deadline)
            expected = dict(
                token=request["_receive_token"],
                frame_index=request["frame_index"],
                files=len(request["files"]),
                bytes=sum(f["bytes"] for f in request["files"].values()),
                payload_bytes=prepared.payload_bytes,
                wire_bytes=prepared.wire_bytes,
                wire_encoding=prepared.wire_encoding,
                hashes_verified=True,
                transport=PROTOCOL,
            )
            if any(result.get(key) != value for key, value in expected.items()):
                raise ValueError("Invalid server persistence receipt")
            result.update(send_timing)
            self._clock_receipt(result, deadline)
            return result
        except Exception:
            self.close()
            raise

    def close(self):
        proc, self.proc = self.proc, None
        self.buffer.clear()
        if proc is None:
            return
        proc.stdin.close()
        try:
            proc.wait(timeout=2)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)
        proc.stdout.close()


class PermanentUploadError(Exception):
    """The server rejected this frame; sending the same request again cannot succeed."""


class _RefuseRedirect(urllib.request.HTTPRedirectHandler):
    # urllib would repeat a redirected POST as a bodiless GET and report success.
    def redirect_request(self, *args, **kwargs):
        return None


class HttpUploader:
    """POST one frame's JPG and JSON to PORTHOLE_API_URL as multipart/form-data.

    Parts are named jpg and json; X-Record-Id carries the JSON record_id so
    the server can drop repeats, and PORTHOLE_API_TOKEN, if set, is sent as a
    Bearer token. 2xx and 409 (already received) count as delivered. Malformed
    requests (400, 413, 415, 422) are permanent; everything else, including
    redirects, counts as a failed send (UploadWorker then drops the frames waiting).
    """

    CONTENT_TYPES = dict(jpg="image/jpeg", json="application/json")
    PERMANENT_STATUS = frozenset((400, 413, 415, 422))

    def __init__(self, options, timeout=60):
        self.options, self.timeout = (options, timeout)
        self.opener = urllib.request.build_opener(_RefuseRedirect)

    def upload(self, source, manifest):
        parts = {}
        validate_manifest(dict(manifest, _receive_token=uuid.uuid4().hex))
        for name, info in manifest["files"].items():
            data = (Path(source) / name).read_bytes()
            if len(data) != info["bytes"] or hashlib.sha256(data).hexdigest() != info["sha256"]:
                raise ValueError(f"Artifact changed after commit: {name}")
            parts[Path(name).suffix.lstrip(".").lower()] = (name, data)
        if set(parts) != set(self.CONTENT_TYPES):
            raise PermanentUploadError("Upload needs exactly one JPG and one JSON")
        record_id = json.loads(parts["json"][1])["record_id"]
        boundary = uuid.uuid4().hex
        body = bytearray()
        for field, content_type in self.CONTENT_TYPES.items():
            name, data = parts[field]
            body += (
                f'--{boundary}\r\nContent-Disposition: form-data; name="{field}"; '
                f'filename="{name}"\r\nContent-Type: {content_type}\r\n\r\n'
            ).encode()
            body += data + b"\r\n"
        body += f"--{boundary}--\r\n".encode()
        headers = {"Content-Type": f"multipart/form-data; boundary={boundary}",
                   "X-Record-Id": record_id}
        if self.options.api_token:
            headers["Authorization"] = f"Bearer {self.options.api_token}"
        request = urllib.request.Request(
            self.options.api_url, data=bytes(body), headers=headers, method="POST"
        )
        started = time.monotonic_ns()
        try:
            with self.opener.open(request, timeout=self.timeout) as response:
                status, text = (response.status, response.read(MAX_HEADER))
        except urllib.error.HTTPError as exc:
            detail = exc.read(512).decode(errors="replace")
            if exc.code in self.PERMANENT_STATUS:
                raise PermanentUploadError(f"HTTP {exc.code}: {detail}") from None
            if exc.code != 409:
                raise RuntimeError(f"API upload failed: HTTP {exc.code}: {detail}") from None
            status, text = (409, b"already received")
        return dict(
            transport="http_multipart_v1", url=self.options.api_url, record_id=record_id,
            http_status=status, response=text.decode(errors="replace")[:2000],
            files=len(parts), bytes=len(body),
            client_send_ms=(time.monotonic_ns() - started) / 1e6,
            completed_epoch_ns=time.time_ns(),
        )

    def close(self):
        pass


def _load_local_env() -> None:
    """Load a sibling .env without overriding explicitly set environment values.

    As in a shell, a later assignment in the file wins over an earlier one.
    """
    env_path = Path(__file__).with_name(".env")
    if not env_path.is_file():
        return
    values = {}
    for raw_line in env_path.read_text(encoding="utf-8-sig").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if not re.fullmatch("[A-Za-z_][A-Za-z0-9_]*", key):
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and (value[0] in "\"'"):
            value = value[1:-1]
        values[key] = value
    for key, value in values.items():
        os.environ.setdefault(key, value)


UPLOAD_SETTINGS = (
    "PORTHOLE_UPLOAD_HOST",
    "PORTHOLE_UPLOAD_USER",
    "PORTHOLE_UPLOAD_DIR",
    "PORTHOLE_UPLOAD_KEY",
)


def terminal_id():
    """Terminal name used in record_id (PORTHOLE_TERMINAL_ID, else the host name)."""
    return os.getenv("PORTHOLE_TERMINAL_ID", "").strip() or socket.gethostname()


def upload_options():
    """Upload settings come only from the environment or the sibling .env.

    PORTHOLE_API_URL selects the HTTP API (JPG/JSON pairs); without it the SSH
    receiver settings are used.
    """
    bw_kib = int(os.getenv("PORTHOLE_UPLOAD_BW_KIB", "24576"))
    if bw_kib <= 0:
        raise ValueError("PORTHOLE_UPLOAD_BW_KIB must be positive")
    api_url = os.getenv("PORTHOLE_API_URL", "").strip()
    if api_url:
        return SimpleNamespace(
            mode="api", api_url=api_url, api_token=os.getenv("PORTHOLE_API_TOKEN", "").strip(),
            bw_kib=bw_kib,
        )
    missing = [name for name in UPLOAD_SETTINGS if not os.getenv(name, "").strip()]
    if missing:
        raise ValueError(
            "upload is not configured; set PORTHOLE_API_URL or " + ", ".join(missing) + " in .env"
        )
    return SimpleNamespace(
        mode="ssh",
        host=os.environ["PORTHOLE_UPLOAD_HOST"].strip(),
        user=os.environ["PORTHOLE_UPLOAD_USER"].strip(),
        destination=os.environ["PORTHOLE_UPLOAD_DIR"].strip(),
        key=Path(os.environ["PORTHOLE_UPLOAD_KEY"].strip()).expanduser(),
        bw_kib=bw_kib,
    )


class UploadWorker:
    """Send reported frames from UPLOAD_STAGING, oldest first, and delete each one after.

    HTTP API when PORTHOLE_API_URL is set, else the embedded byte/hash-verified SSH
    transport. Nothing is kept for later: when a send fails (no network, server down)
    the frames waiting are dropped and, for OFFLINE_RETRY_SEC, new detections are only
    listed in the CSV. A frame the server rejects is dropped as well.
    """

    def __init__(self, timeout=30):
        self.timeout = timeout
        self.jobs = queue.Queue()
        self.offline_until = 0.0
        self.stop_event = threading.Event()
        self.error = ""
        self.busy = False  # a frame is being sent
        self.thread = threading.Thread(target=self.run, name="live-artifact-upload", daemon=True)
        try:
            self.options = upload_options()
        except ValueError as exc:
            self.enabled, self.error = (False, str(exc))
            print(f"[UPLOAD] off, detections are only listed in the CSV: {exc}", flush=True)
            return
        self.enabled = True
        self.thread.start()

    def accepting(self):
        """Whether a newly reported frame should be prepared for sending."""
        return (self.enabled and time.monotonic() >= self.offline_until
                and self.jobs.qsize() < UPLOAD_QUEUE_LIMIT)

    def send(self, key, index, folder, manifest):
        self.jobs.put((key, index, Path(folder), manifest))

    def pending(self):
        return self.jobs.qsize()

    def waiting(self):
        """Whether a frame is being sent or waits to be (RawUploader holds back meanwhile)."""
        return self.busy or not self.jobs.empty()

    def drop_waiting(self):
        while True:
            try:
                folder = self.jobs.get_nowait()[2]
            except queue.Empty:
                return
            shutil.rmtree(folder, ignore_errors=True)

    def run(self):
        uploader, active_key = (None, None)
        options = self.options
        try:
            target = options.api_url if options.mode == "api" else f"{options.user}@{options.host}"
            print(f"[UPLOAD] {options.mode} -> {target}", flush=True)
            while not self.stop_event.is_set():
                try:
                    key, index, folder, manifest = self.jobs.get(timeout=1)
                except queue.Empty:
                    continue
                if time.monotonic() < self.offline_until:  # queued just as a send failed
                    shutil.rmtree(folder, ignore_errors=True)
                    continue
                self.busy = True
                try:
                    if key != active_key:
                        if uploader:
                            uploader.close()
                        uploader = (
                            HttpUploader(options, timeout=self.timeout)
                            if options.mode == "api"
                            else PersistentUploader(
                                options, Path(REMOTE_FOLDER) / key / "certifcate", timeout=self.timeout
                            )
                        )
                        active_key = key
                    receipt = uploader.upload(folder / "certifcate", manifest)
                    self.error = ""
                    print(f"[UPLOAD] verified run={key} frame={index} files={receipt['files']}", flush=True)
                except PermanentUploadError as exc:
                    print(f"[UPLOAD] rejected by server, dropped run={key} frame={index}: {exc}", flush=True)
                except Exception as exc:
                    self.error = str(exc)
                    self.offline_until = time.monotonic() + OFFLINE_RETRY_SEC
                    dropped = 1 + self.jobs.qsize()
                    self.drop_waiting()
                    print(f"[UPLOAD] cannot send ({exc}); dropped {dropped} frame(s), "
                          f"CSV only for {OFFLINE_RETRY_SEC} s", flush=True)
                    if uploader:
                        uploader.close()
                    uploader, active_key = (None, None)
                finally:
                    shutil.rmtree(folder, ignore_errors=True)
                    self.busy = False
        except Exception as exc:
            self.enabled, self.error = (False, str(exc))
            print(f"[UPLOAD] worker stopped, detections are only listed in the CSV: {exc}", flush=True)
        finally:
            self.drop_waiting()
            if uploader:
                uploader.close()

    def close(self, drain_seconds=3):
        """Give waiting frames a moment, then stop; the whole shutdown has to fit in the 15 s the
        analysis terminal allows before it kills the process (the CSV rows are already on disk)."""
        deadline = time.monotonic() + drain_seconds
        while self.jobs.qsize() and self.thread.is_alive() and time.monotonic() < deadline:
            time.sleep(0.1)
        self.stop_event.set()
        if self.thread.is_alive():
            self.thread.join(timeout=5)
        self.drop_waiting()


RAW_POLL_SEC = 60  # a run is finished every 10 minutes


RAW_SETTLE_SEC = 60


RAW_RETRY_MAX_SEC = 3600  # a run that keeps failing is tried again after 1, 2, 4 ... minutes, at most this


RAW_UNREACHABLE = frozenset((5, 10, 12, 30, 35, 255))


def raw_upload_options():
    """Where the recordings go: PORTHOLE_RAW_DIR on the server of the SSH upload settings
    (PORTHOLE_UPLOAD_HOST, _USER, _KEY, also when PORTHOLE_API_URL sends the detections).
    Only with PORTHOLE_RAW_UPLOAD=true; unset or false, nothing is sent."""
    switch = os.getenv("PORTHOLE_RAW_UPLOAD", "").strip()
    if switch.lower() != "true":
        raise ValueError(f"PORTHOLE_RAW_UPLOAD={switch or 'false'} (true in .env sends them)")
    names = ("PORTHOLE_RAW_DIR", "PORTHOLE_UPLOAD_HOST", "PORTHOLE_UPLOAD_USER", "PORTHOLE_UPLOAD_KEY")
    missing = [name for name in names if not os.getenv(name, "").strip()]
    if missing:
        raise ValueError("set " + ", ".join(missing) + " in .env")
    bw_kib = int(os.getenv("PORTHOLE_UPLOAD_BW_KIB", "24576"))
    if bw_kib <= 0:
        raise ValueError("PORTHOLE_UPLOAD_BW_KIB must be positive")
    return SimpleNamespace(
        destination=os.environ["PORTHOLE_RAW_DIR"].strip().rstrip("/"),
        host=os.environ["PORTHOLE_UPLOAD_HOST"].strip(),
        user=os.environ["PORTHOLE_UPLOAD_USER"].strip(),
        key=Path(os.environ["PORTHOLE_UPLOAD_KEY"].strip()).expanduser(),
        bw_kib=bw_kib,
    )


def collector_busy(held):
    """Whether the recorder's queue is under pressure, as wait_for_collector judges it (held:
    the caller already holds back, which lowers the level at which it may go on)."""
    try:
        row = json.loads(Path("/dev/shm/porthole_collector_pressure.json").read_text())
        os.kill(int(row["pid"]), 0)
        stale = time.time() - float(row["updated_at"]) > 2
        pressure = int(row.get("queue_pending", 0))
        return bool(row.get("active", True)) and (stale or pressure >= (9 if held else 32))
    except (OSError, ValueError, KeyError, json.JSONDecodeError):
        return False


class RawUploader:
    """Send every finished recording to <PORTHOLE_RAW_DIR>/<date>/<run>/ on the server, as it is
    on the SSD except PCAP files, oldest first, and the date folders' CSVs whenever they change.

    Started with the analysis of the first new frame, so only once the collector records with
    PTP ready; a run in which nothing was recorded (closed before that) is not sent. rsync resumes
    a run cut short and leaves out the files the server already has; runs sent completely are
    listed in RAW_DONE and not looked at again. A run that fails on its own is tried again later
    (RAW_RETRY_MAX_SEC) while the others go on. rsync is held while live detections wait to be
    sent or the recorder's queue is under pressure, runs at the analysis' low CPU and disk
    priority, and is killed with this process (setpriv --pdeathsig), held or not.
    """

    def __init__(self, root, detections):
        self.root, self.detections = Path(root), detections
        self.stop_event = threading.Event()
        self.unreachable = False
        self.thread = threading.Thread(target=self.run, name="raw-upload", daemon=True)
        try:
            self.options = raw_upload_options()
        except ValueError as exc:
            print(f"[RAW] recordings are not sent: {exc}", flush=True)
            return
        self.thread.start()

    def finished_runs(self, done):
        """(key, path) of the finished recordings not sent yet, oldest first, once settled."""
        for run in discover(self.root):
            key = run.relative_to(self.root).as_posix()
            if key in done:
                continue
            meta = run / "meta/run_meta.jsonl"
            try:
                if time.time() - meta.stat().st_mtime < RAW_SETTLE_SEC:
                    continue
                with meta.open("rb") as stream:
                    finished = any(json.loads(line).get("event") == "run_finished"
                                   for line in stream if line.endswith(b"\n") and line.strip())
            except (OSError, ValueError):
                finished = False
            if finished:
                yield key, run

    @staticmethod
    def recorded(run):
        """Whether the collector saved camera frames or LiDAR files in the run."""
        return any((run / name).is_file() and (run / name).stat().st_size
                   for name in ("frames/frames.jsonl", "lidar/pcaps.jsonl"))

    def rsync(self, source):
        """rsync one path (<root>/./<date>/...) to the server folder, held while it should wait:
        (exit status, GB sent, last error line), or (None, None, "") when stopping."""
        ssh = shlex.join(["ssh", "-i", str(self.options.key), "-o", "BatchMode=yes",
                          "-o", "StrictHostKeyChecking=yes", "-o", "ConnectTimeout=10",
                          "-o", "ServerAliveInterval=15", "-o", "ServerAliveCountMax=4"])
        command = ["setpriv", "--pdeathsig", "KILL",
                   "rsync", "-rt", "--omit-dir-times", "--relative", "--partial-dir=.rsync-partial",
                   "--exclude=*.[pP][cC][aA][pP]",  # Preserve old local PCAPs, but never send them.
                   "--timeout=600", f"--bwlimit={self.options.bw_kib}", "--stats", "-e", ssh, source,
                   f"{self.options.user}@{self.options.host}:{self.options.destination}/"]
        with tempfile.TemporaryFile("w+") as output:
            proc = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=output,
                                    stderr=subprocess.STDOUT, text=True)
            held = False
            try:
                while proc.poll() is None:
                    if self.stop_event.is_set():
                        return None, None, ""
                    hold = (self.detections is not None and self.detections.waiting()) or collector_busy(held)
                    if hold != held:
                        proc.send_signal(signal.SIGSTOP if hold else signal.SIGCONT)
                        held = hold
                    self.stop_event.wait(0.5)
            finally:
                if proc.poll() is None:
                    proc.send_signal(signal.SIGCONT)
                    proc.terminate()
                    try:
                        proc.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                        proc.wait()
            output.seek(0)
            text = output.read()
        sent = re.search(r"Total bytes sent: ([\d,]+)", text)
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        error = "" if proc.returncode == 0 else next(
            (line for line in reversed(lines) if line.startswith(("rsync", "ssh"))), lines[-1] if lines else "")
        return proc.returncode, None if sent is None else int(sent.group(1).replace(",", "")) / 1e9, error

    def reach(self, status, error):
        """Note whether the server answered (status of an rsync); False when it cannot be reached."""
        if status in RAW_UNREACHABLE:
            if not self.unreachable:
                print(f"[RAW] server not reachable (rsync exit {status}: {error}); "
                      f"trying again every {RAW_POLL_SEC} s", flush=True)
            self.unreachable = True
            return False
        if self.unreachable:
            print("[RAW] server reachable again", flush=True)
        self.unreachable = False
        return True

    def send_runs(self, done, failures, retry_at):
        """Send the finished runs not sent yet; False when the server cannot be reached."""
        for key, run in self.finished_runs(done):
            if self.stop_event.is_set():
                return False
            if time.monotonic() < retry_at.get(key, 0):
                continue
            if self.recorded(run):
                started = time.monotonic()
                status, gb, error = self.rsync(f"{self.root}/./{key}")
                if status is None or not self.reach(status, error):
                    return False
                if status:
                    failures[key] = failures.get(key, 0) + 1
                    wait = min(RAW_POLL_SEC * 2 ** (failures[key] - 1), RAW_RETRY_MAX_SEC)
                    retry_at[key] = time.monotonic() + wait
                    print(f"[RAW] {key} not sent completely (rsync exit {status}: {error}); "
                          f"trying it again in {wait:.0f} s", flush=True)
                    continue
                print(f"[RAW] sent {key} in {time.monotonic() - started:.0f} s"
                      + ("" if gb is None else f" ({gb:.2f} GB)"), flush=True)
            with RAW_DONE.open("a") as stream:
                stream.write(key + "\n")
                stream.flush()
                os.fsync(stream.fileno())
            done.add(key)
        return True

    def send_csvs(self, done, sent):
        """The date folders' CSVs of the sent runs, again whenever one changes (the analysis can
        list detections after their run was sent)."""
        for date in sorted({key.split("/")[0] for key in done}):
            try:
                stat = (self.root / date / DETECTIONS_CSV).stat()
            except FileNotFoundError:
                continue
            if sent.get(date) == (stat.st_size, stat.st_mtime_ns):
                continue
            status, _, error = self.rsync(f"{self.root}/./{date}/{DETECTIONS_CSV}")
            if status is None or not self.reach(status, error):
                return
            if status == 0:
                sent[date] = (stat.st_size, stat.st_mtime_ns)

    def run(self):
        import fcntl

        print(f"[RAW] recordings -> {self.options.user}@{self.options.host}:"
              f"{self.options.destination}/<date>/<run>", flush=True)
        try:
            done = set(RAW_DONE.read_text().split())
        except OSError:
            done = set()
        RAW_DONE.parent.mkdir(exist_ok=True)
        failures, retry_at, sent_csv = {}, {}, {}
        with RAW_LOCK.open("a") as lock:
            while not self.stop_event.is_set():
                try:
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:  # another sender (e.g. one started by hand) is running
                    self.stop_event.wait(RAW_POLL_SEC)
                    continue
                try:
                    if self.send_runs(done, failures, retry_at):
                        self.send_csvs(done, sent_csv)
                except Exception as exc:  # e.g. a run deleted meanwhile: look again later
                    print(f"[RAW] {type(exc).__name__}: {exc}; trying again in {RAW_POLL_SEC} s", flush=True)
                finally:
                    fcntl.flock(lock, fcntl.LOCK_UN)
                self.stop_event.wait(RAW_POLL_SEC)

    def close(self):
        self.stop_event.set()
        if self.thread.is_alive():
            self.thread.join(timeout=6)

SERVER_TRANSPORT_SOURCE = server_transport_source()

def discover(root):
    """Target adapter: only explicitly closed ten-minute sessions, not old minute folders."""
    return sorted(path.parent.parent for path in Path(root).glob("*/*/meta/run_meta.jsonl")
                  if re.fullmatch(r"\d{8}_\d{3}[0]", path.parent.parent.name))


class RawSessionUploader(RawUploader):
    @staticmethod
    def recorded(run):
        # Target has individual files and GPS/detection logs, not PTP manifests.
        return any(path.is_file() and path.stat().st_size
                   for name in ("frames", "frames_bbox", "meta", "lidar", "gps", "logs")
                   for path in (run / name).glob("*") if path.name != "run_meta.jsonl")


def stage_saved_frame(uploader, base_dir, file_base, frame_id):
    """Copy only a committed JPG/JSON pair; never give originals to the dropper."""
    if uploader is None or not uploader.accepting():
        return
    folder = None
    try:
        artifacts = [base_dir / sub / (file_base + ext)
                     for sub, ext in (("frames", ".jpg"), ("meta", ".json"))]
        files = {path.name: dict(bytes=path.stat().st_size, sha256=sha(path)) for path in artifacts}
        reason = artifact_pair_error(files)
        if reason:
            raise ValueError(reason)
        key = base_dir.parent.name + "/" + base_dir.name
        staging = UPLOAD_STAGING / key
        staging.mkdir(parents=True, exist_ok=True)
        folder = Path(tempfile.mkdtemp(prefix=f"{frame_id:08d}_", dir=staging))
        certificate = folder / "certifcate"  # Keep the source transport's literal spelling.
        certificate.mkdir()
        for path in artifacts:
            shutil.copyfile(path, certificate / path.name)
        uploader.send(key, frame_id, folder, dict(frame_index=frame_id, files=files, frame_log={}))
    except Exception as exc:
        print(f"[UPLOAD] not sent; local files retained: {type(exc).__name__}: {exc}", flush=True)
        if folder is not None:
            shutil.rmtree(folder, ignore_errors=True)


class SessionUploads:
    """Target-only writer completion adapter; wire formats/retries remain the source's.

    Main saver and AI logger share this short file-I/O lock, never a network call.
    GPS owns its old session until its handle is closed. No wall-clock timeout can
    label that still-open log finished; shutdown also checks the writer threads.
    """

    def __init__(self, root="/media/hucomputer/DISK/pothole_runs"):
        self.root = Path(root)
        self.lock = threading.RLock()
        self.sessions = set()
        self.detections = self.raw = None

    def start(self):
        _load_local_env()
        shutil.rmtree(UPLOAD_STAGING, ignore_errors=True)  # Only this app's disposable copies.
        self.detections = UploadWorker()
        self.raw = RawSessionUploader(self.root, self.detections)

    def track(self, folder):
        with self.lock:
            folder = Path(folder)
            self.sessions.add(folder)
            # A restart in the same bucket is active again, not a completed recording.
            (folder / "meta/run_meta.jsonl").unlink(missing_ok=True)

    def log(self, detections):
        with self.lock:
            append_detection_log(detections, root_dir=str(self.root), session_uploads=self)

    def finish_sessions(self, gps_recorder, stopping=False):
        with self.lock:
            now = datetime.now()
            current = now.replace(minute=now.minute // 10 * 10, second=0, microsecond=0).strftime("%Y%m%d_%H%M")
            active = set()
            if gps_recorder is not None and not stopping:
                for session in (gps_recorder.recorder, gps_recorder.pending_recorder):
                    if session is not None:
                        active.add(session.gps_jsonl.parent.parent)
            for folder in sorted(self.sessions):
                if not stopping and (folder.name >= current or folder in active):
                    continue
                meta = folder / "meta/run_meta.jsonl"
                meta.parent.mkdir(parents=True, exist_ok=True)
                with meta.open("w", encoding="utf-8") as stream:
                    stream.write(json.dumps(dict(event="run_finished")) + "\n")
                    stream.flush()
                    os.fsync(stream.fileno())
                self.sessions.remove(folder)

    def close(self, gps_recorder, writers_stopped):
        try:
            gps_thread = None if gps_recorder is None else gps_recorder.thread
            if writers_stopped and (gps_thread is None or not gps_thread.is_alive()):
                if gps_recorder is not None:
                    # GPS may consume its reservation before main refreshes the session.
                    self.track(gps_recorder.recorder.gps_jsonl.parent.parent)
                self.finish_sessions(gps_recorder, stopping=True)
        finally:
            try:
                if self.raw is not None:
                    self.raw.close()
            finally:
                if self.detections is not None:
                    self.detections.close()

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

    # systemctl stop(SIGTERM) 시에도 finally가 실행되어 GPS 로그가 정상 마감되도록 함
    def _handle_stop_signal(signum, frame):
        print(f"[SYS] 종료 신호 수신({signum}), 정리 후 종료합니다.", flush=True)
        stop_event.set()
    signal.signal(signal.SIGTERM, _handle_stop_signal)
    signal.signal(signal.SIGINT, _handle_stop_signal)
    print(f"[SYS] 화면 출력: {'ON' if SHOW_WINDOW else 'OFF (headless)'}", flush=True)

    main_reader = FFmpegStreamReader(main_url, W_MAIN, H_MAIN, main_buffer, is_sub=False)
    main_reader.start()

    sub_reader = FFmpegStreamReader(sub_url, W_SUB, H_SUB, sub_buffer, is_sub=True)
    sub_reader.start()

    scale_x = W_MAIN / W_SUB
    scale_y = H_MAIN / H_SUB

    # 2개 모델 초기화 (스크립트 위치 기준 절대경로)
    model_dir = os.path.dirname(os.path.abspath(__file__))
    ai_pothole = DeepXPPUModel(engine_path=os.path.join(model_dir, "pothole_best_ppu.dxnn"), conf_thres=0.2)
    ai_roadobj = DeepXPPUModel(engine_path=os.path.join(model_dir, "roadobj_ppu.dxnn"), conf_thres=0.2)

    uploads = SessionUploads()
    ai_thread = threading.Thread(
        target=ai_worker_loop,
        args=(ai_pothole, ai_roadobj, sub_buffer, result_buffer, stop_event, scale_x, scale_y, uploads),
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
        uploads.track(gps_session.gps_jsonl.parent.parent)
        uploads.start()
        last_gps_refresh = 0.0
        last_gps_report = 0.0
        while not stop_event.is_set():
            now_mono = time.monotonic()
            if now_mono - last_gps_refresh >= 0.2:
                try:
                    gps_session = refresh_gps_session(gps_session, gps_recorder, datetime.now())
                    uploads.track(gps_session.gps_jsonl.parent.parent)
                    uploads.finish_sessions(gps_recorder)
                except Exception as e:
                    print(f"[GPS WARN] refresh 실패: {e}", flush=True)
                last_gps_refresh = now_mono
            if now_mono - last_gps_report >= 10.0:  # 10초마다 GPS 상태 출력
                st = gps_recorder.stats()
                print(f"[GPS] connected={gps_recorder.is_connected()} device={st['active_device']} "
                      f"sentences={st['sentence_count']} valid_fix={st['valid_fix_count']} "
                      f"last_err={st['transport_errors'][-1:] or '-'}", flush=True)
                last_gps_report = now_mono
            frame_item = main_buffer.get_and_clear()
            if frame_item is None:
                time.sleep(0.005)
                continue
            frame_main, frame_timestamp = frame_item

            main_frame_count += 1
            current_detections = result_buffer.get_current() or []

            if current_detections and current_detections != last_saved_detections:
                try:
                    with uploads.lock:
                        gps_session = save_detection_data(frame_main.copy(), current_detections, main_frame_count, terminal_id=terminal_id(), frame_timestamp=frame_timestamp, gps_session=gps_session, gps_recorder=gps_recorder, session_uploads=uploads)
                except Exception as e:
                    print(f"[SAVE WARN] 저장 실패, 계속 진행: {e}", flush=True)
                last_saved_detections = current_detections

            if not SHOW_WINDOW:
                continue

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
            try:
                ai_thread.join(timeout=5)
                uploads.close(gps_recorder, writers_stopped=not ai_thread.is_alive())
            finally:
                if SHOW_WINDOW:
                    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
