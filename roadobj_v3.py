import os
import cv2
import numpy as np
import subprocess
import threading
import time
import json
from datetime import datetime, timezone
import queue
from dx_engine import InferenceEngine, InferenceOption

# --- [1] 타겟 클래스 설정 ---
TARGET_CLASSES = {
    4: 'Speed_Bump',
    5: 'Traffic_Signal',
    8: 'Street_Name_Plate',
    10: 'CCTV',
    12: 'Horizontal_Member'
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

# --- [3] 딥엑스 PPU 추론 클래스 ---
class DeepXRoadObjectDetectorPPU:
    def __init__(self, engine_path, conf_thres=0.4, iou_thres=0.45):
        self.engine_path = engine_path
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.input_width = 640
        self.input_height = 640
        self.input_layout = "hwc"
        self.input_has_batch = False

        self.engine_pool = queue.Queue(maxsize=1)
        io = InferenceOption()
        engine = InferenceEngine(self.engine_path, io)
        self.engine_pool.put(engine)
        
        self._load_input_shape(engine)
        print(f"[AI] DeepX PPU 엔진 로드 완료: {engine_path}")

    def _load_input_shape(self, engine):
        try:
            input_info = engine.get_input_tensors_info()
            shape = list(input_info[0].get("shape", []))
        except Exception as e:
            print(f"[DeepX] 입력 텐서 shape 확인 실패: {e}")
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

    def letter_box(self, img):
        h, w = img.shape[:2]
        scale = min(self.input_width / w, self.input_height / h)
        nw, nh = int(w * scale), int(h * scale)

        resized = cv2.resize(img, (nw, nh))
        canvas = np.full((self.input_height, self.input_width, 3), 114, dtype=np.uint8)

        dw, dh = (self.input_width - nw) // 2, (self.input_height - nh) // 2
        canvas[dh:dh+nh, dw:dw+nw] = resized
        return canvas, scale, (dw, dh)

    def _prepare_input_tensor(self, npu_input):
        input_tensor = cv2.cvtColor(npu_input, cv2.COLOR_BGR2RGB)
        if self.input_layout in ["nchw", "chw"]:
            input_tensor = np.transpose(input_tensor, (2, 0, 1))
        if self.input_has_batch:
            input_tensor = np.expand_dims(input_tensor, axis=0)
        return np.ascontiguousarray(input_tensor, dtype=np.uint8)

    def infer(self, img):
        if img is None: return []

        h_orig, w_orig = img.shape[:2]
        npu_input, scale, offset = self.letter_box(img)
        input_tensor = self._prepare_input_tensor(npu_input)

        engine = self.engine_pool.get()
        try:
            output_tensor = engine.run([input_tensor])
            raw_dets = self.postprocess_ppu(output_tensor, self.conf_thres, self.iou_thres)
            
            res = []
            dw, dh = offset
            for box, score, cls_id in raw_dets:
                if cls_id not in TARGET_CLASSES:
                    continue
                    
                x1 = np.clip((box[0] - dw) / scale, 0, w_orig)
                y1 = np.clip((box[1] - dh) / scale, 0, h_orig)
                x2 = np.clip((box[2] - dw) / scale, 0, w_orig)
                y2 = np.clip((box[3] - dh) / scale, 0, h_orig)
                res.append([int(x1), int(y1), int(x2), int(y2), float(score), int(cls_id)])
                
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

# --- [4] 데이터 저장 도우미 함수 ---
def save_detection_data(frame, detections, frame_id, terminal_id="terminal01", root_dir="/mnt/ssd/porthole_runs"):
    """탐지 결과를 명세서에 맞게 구조화하여 디스크에 저장합니다."""
    # 시간 정보 생성[cite: 6, 7]
    now_kst = datetime.now()
    now_utc = datetime.now(timezone.utc)
    
    date_str = now_kst.strftime("%Y%m%d")
    session_str = now_kst.strftime("%Y%m%d_%H%M")
    
    # 폴더 구조 생성 (tree 명세서 반영)[cite: 7]
    base_dir = os.path.join(root_dir, date_str, session_str)
    frames_dir = os.path.join(base_dir, "frames")
    gps_dir = os.path.join(base_dir, "gps")
    lidar_dir = os.path.join(base_dir, "lidar")
    meta_dir = os.path.join(base_dir, "meta")
    
    for d in [frames_dir, gps_dir, lidar_dir, meta_dir]:
        os.makedirs(d, exist_ok=True)
        
    # 고유 식별자 및 파일명 규칙 적용[cite: 6]
    timestamp_ns = time.time_ns()
    file_base = f"frame_{timestamp_ns}_{frame_id:08d}"
    
    jpg_filename = f"{file_base}.jpg"
    json_filename = f"{file_base}.json"
    pcap_filename = f"{file_base}.pcap"
    
    # JSON 객체 조립 (명세서 항목 준수)[cite: 6]
    record_id = f"{terminal_id}/{session_str}/{frame_id}"
    
    unique_classes = set(cls_id for _, _, _, _, _, cls_id in detections)
    categories = [{"id": int(cid), "name": TARGET_CLASSES[cid]} for cid in unique_classes]
    
    images = [{
        "id": 1,
        "width": frame.shape[1],
        "height": frame.shape[0],
        "file_name": jpg_filename,
        "date_captured": now_utc.strftime("%Y-%m-%dT%H:%M:%SZ") # UTC 표기 강제[cite: 6]
    }]
    
    annotations = []
    for i, (x1, y1, x2, y2, score, cls_id) in enumerate(detections):
        w = x2 - x1
        h = y2 - y1
        annotations.append({
            "id": i + 1,
            "image_id": 1,
            "category_id": int(cls_id),
            "confidence": round(float(score), 4),
            "bbox": [int(x1), int(y1), int(w), int(h)], # x, y, 너비, 높이[cite: 6]
            "segmentation": [[int(x1), int(y1), int(x2), int(y1), int(x2), int(y2), int(x1), int(y2)]],
            "measurements": {
                "size": {"length_m": None, "width_m": None, "area_m2": None}, # 라이다 전용 값 null 처리[cite: 6]
                "depth": {"median_cm": None}
            }
        })
        
    data = {
        "record_id": record_id,
        "categories": categories,
        "images": images,
        "annotations": annotations,
        "gps": {"latitude_deg": None, "longitude_deg": None}, # GPS 미연동 시 null[cite: 6]
        "lidar": {"pcap_files": [{"name": pcap_filename}]}
    }
    
    # 1. 이미지 저장 (frames/)[cite: 6, 7]
    cv2.imwrite(os.path.join(frames_dir, jpg_filename), frame)
    
    # 2. JSON 정보 저장 (meta/)[cite: 6, 7]
    with open(os.path.join(meta_dir, json_filename), 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        
    # 3. 빈 PCAP 파일 생성 (lidar/)[cite: 6, 7]
    with open(os.path.join(lidar_dir, pcap_filename), 'wb') as f:
        pass

# --- [5] AI 백그라운드 워커 스레드 ---
def ai_worker_loop(detector, frame_buffer, result_buffer, stop_event):
    while not stop_event.is_set():
        frame = frame_buffer.get_and_clear()
        if frame is None:
            time.sleep(0.005)
            continue
            
        detections = detector.infer(frame)
        result_buffer.put(detections)

def read_exact(pipe, size):
    data = bytearray(size)
    view = memoryview(data)
    bytes_read = 0
    while bytes_read < size:
        chunk = pipe.stdout.read(size - bytes_read)
        if not chunk: 
            return None
        view[bytes_read:bytes_read+len(chunk)] = chunk
        bytes_read += len(chunk)
    return bytes(data)

# --- [6] 메인 파이프라인 ---
def main():
    w, h = 1920, 1080
    rtsp_url = "rtsp://admin:Hu924688@192.168.11.64:554/Streaming/Channels/101"
    
    frame_buffer = LatestItemBuffer()
    result_buffer = LatestItemBuffer()
    stop_event = threading.Event()

    ai_detector = DeepXRoadObjectDetectorPPU(engine_path="roadobj_ppu.dxnn", conf_thres=0.5)

    ai_thread = threading.Thread(
        target=ai_worker_loop, 
        args=(ai_detector, frame_buffer, result_buffer, stop_event), 
        daemon=True
    )
    ai_thread.start()

    command = [
        'ffmpeg',
        '-nostdin',                   
        '-hwaccel', 'drm',            
        '-rtsp_transport', 'tcp',     
        '-fflags', 'nobuffer',        
        '-flags', 'low_delay',        
        '-i', rtsp_url,               
        '-f', 'image2pipe',           
        '-pix_fmt', 'nv12',           
        '-vcodec', 'rawvideo',        
        '-'                           
    ]

    print("FFmpeg DRM 디코딩 (비동기 AI 추론 및 데이터 저장) 시작...")
    
    frame_size = int(w * h * 1.5)
    pipe = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=10**8)

    frame_id = 0
    try:
        while True:
            raw_data = read_exact(pipe, frame_size)
            if not raw_data:
                print("FFmpeg 스트림이 종료되었습니다.")
                break
            
            frame_id += 1
            yuv_img = np.frombuffer(raw_data, dtype='uint8').reshape((int(h * 1.5), w))
            frame = cv2.cvtColor(yuv_img, cv2.COLOR_YUV2BGR_NV12)
            
            frame_buffer.put(frame.copy())
            detections = result_buffer.get_current() or []
            
            # AI 결과가 탐지되었을 때 디스크에 트리 구조로 파일 생성[cite: 6, 7]
            if len(detections) > 0:
                # 백그라운드 스레드에서 I/O를 수행하도록 던지는 것이 더 좋으나, 
                # 직관성을 위해 여기서 저장 함수를 호출합니다.
                save_detection_data(frame, detections, frame_id)
            
            for x1, y1, x2, y2, score, cls_id in detections:
                label = f"{TARGET_CLASSES[cls_id]} ({score:.2f})"
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, max(20, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            cv2.imshow('RPi5 Async HW Decode + DeepX PPU (Save Active)', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except Exception as e:
        print(f"오류 발생: {e}")

    finally:
        stop_event.set()
        pipe.terminate()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
