import os
import subprocess
import requests
from datetime import datetime

# 라즈베리파이 작업 경로
WORK_DIR = "/home/hucomputer/pothole_rasp"
HF_API_URL = "https://huggingface.co/api/models/HudatersU/pothole_rasp"
HF_MODELS = [
    "pothole_best_ppu.dxnn",
    "roadobj_ppu.dxnn"
]
TAILSCALE_IP = "100.75.164.59"

def update_github():
    print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] --- GitHub 업데이트 확인 ---")
    try:
        os.chdir(WORK_DIR)
        # origin 정보 갱신
        subprocess.run(["git", "fetch", "origin", "main"], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        # 로컬과 리모트 해시(버전) 비교
        local_hash = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("utf-8").strip()
        remote_hash = subprocess.check_output(["git", "rev-parse", "origin/main"]).decode("utf-8").strip()
        
        if local_hash != remote_hash:
            print("새로운 GitHub 업데이트 발견! Pull을 진행합니다.")
            subprocess.run(["git", "pull", "origin", "main"], check=True)
            print("GitHub 동기화 완료.")
        else:
            print("GitHub 리포지토리가 이미 최신 상태입니다.")
    except Exception as e:
        print(f"GitHub 업데이트 중 오류 발생: {e}")

def update_huggingface():
    print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] --- Hugging Face 모델 업데이트 확인 ---")
    try:
        # 허깅페이스 API를 통해 최신 커밋 해시(sha) 확인
        resp = requests.get(HF_API_URL, timeout=10)
        resp.raise_for_status()
        latest_sha = resp.json().get("sha")
    except Exception as e:
        print(f"Hugging Face API 호출 실패 (네트워크 확인 필요): {e}")
        return

    sha_file = os.path.join(WORK_DIR, ".hf_latest_sha")
    local_sha = ""
    
    if os.path.exists(sha_file):
        with open(sha_file, "r") as f:
            local_sha = f.read().strip()

    if latest_sha != local_sha:
        print(f"새로운 모델 커밋 발견! ({latest_sha[:7]}) 다운로드를 시작합니다.")
        for model in HF_MODELS:
            url = f"https://huggingface.co/HudatersU/pothole_rasp/resolve/main/{model}"
            model_path = os.path.join(WORK_DIR, model)
            print(f"- 다운로드 중: {model}")
            
            # 기존 모델 덮어쓰기 다운로드
            subprocess.run(["wget", "-q", "-O", model_path, url], check=True)
            
        # 성공 시 로컬 해시 파일 갱신
        with open(sha_file, "w") as f:
            f.write(latest_sha)
        print("모든 모델 업데이트 완료.")
    else:
        print("Hugging Face 모델이 이미 최신 상태입니다.")

def check_tailscale():
    print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] --- Tailscale 네트워크 상태 점검 ---")
    try:
        # tailscale0 네트워크 인터페이스 정보 확인
        result = subprocess.run(["ip", "addr", "show", "tailscale0"], capture_output=True, text=True)
        
        if TAILSCALE_IP not in result.stdout:
            print(f"Tailscale 연결 유실 (IP: {TAILSCALE_IP} 미발견). 재접속 시도...")
            # Tailscale 재접속 강제 실행
            subprocess.run(["sudo", "tailscale", "up"], check=True)
            print("Tailscale 서비스 재시작 명령을 전송했습니다.")
        else:
            print(f"Tailscale 정상 동작 중 (IP: {TAILSCALE_IP}).")
    except Exception as e:
        print(f"Tailscale 인터페이스 확인 중 오류 (서비스 중단 의심). 재접속 시도...")
        subprocess.run(["sudo", "tailscale", "up"])

if __name__ == "__main__":
    update_github()
    update_huggingface()
    check_tailscale()
    print("-" * 50)
