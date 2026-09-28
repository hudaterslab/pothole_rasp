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

# --- [1] 타겟 클래스 매핑 (2개 모델 분리 및 클래스 한정) ---
# 형식: YOLO_Index: ('JSON_영문명', '화면_출력용_한글명', 원본_Category_ID)

# 1. pothole_best_ppu.dxnn 모델 (빗물받이 전용)
POTHOLE_CLASSES = {
    3: ('Sewer_Road', '빗물받이', 2) # 원본 매핑의 가로재(2) ID를 재사용
}

# 2. roadobj_ppu.dxnn 모델 (도로 시설물 5종 전용)
ROADOBJ_CLASSES = {
    3: ('Road_Mirror', '도로반사경', 10),
    4: ('Speed_Bump', '과속방지턱', 11),
    5: ('Traffic_Signal', '교통신호기', 27),
    8: ('Street_Name_Plate', '도로명판', 30),
    10: ('CCTV', '감시카메라(CCTV)', 32)
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

# --- [3] GPS 연동 더미 함수 ---
def get_current_gps_fix():
    """GPS 장비 연결 시 실제 위/경도 수신 로직으로 변경하세요."""
    # return (37.5665, 126.9780)
    return None, None

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
def save_detection_data(frame, detections, frame_id, terminal_id="terminal01", root_dir="/mnt/ssd/porthole_runs"):
    now_kst = datetime.now()
    now_utc = datetime.now(timezone.utc)
    
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
    
    lat, lon = get_current_gps_fix()
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
        
    categories = [{"id": cid, "name": name} for cid, name in categories_dict.items()]
    
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
                
                frame_id += 1
                if self.is_sub and frame_id % 15 != 0:
                    continue

                yuv = np.frombuffer(raw_data, dtype='uint8').reshape((int(self.h * 1.5), self.w))
                bgr = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_NV12)
                self.buffer.put(bgr.copy())
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
    
    try:
        while True:
            frame_main = main_buffer.get_and_clear()
            if frame_main is None:
                time.sleep(0.005)
                continue
                
            main_frame_count += 1
            current_detections = result_buffer.get_current() or []
            
            if current_detections and current_detections != last_saved_detections:
                save_detection_data(frame_main.copy(), current_detections, main_frame_count)
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
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
