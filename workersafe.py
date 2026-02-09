import subprocess
import logging
import os
import sys
from multiprocessing import Process, Value
from datetime import datetime

# 1. 공통 경로 설정
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

# [추가] 로그 전용 폴더 생성
LOG_DIR = os.path.join(BASE_DIR, "logs")
os.makedirs(LOG_DIR, exist_ok=True)

# 2. 로깅 설정 (logs 폴더 내부에 저장)
def setup_logger(name):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    # 중복 핸들러 방지
    if not logger.handlers:
        # 파일명에 실행 시점의 날짜와 시간을 포함 (예: logs/log_SCA_Algorithm_20260203_1150.log)
        now = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_filename = os.path.join(LOG_DIR, f"log_{name}_{now}.log")
        
        # utf-8-sig: 윈도우 메모장/엑셀에서 한글 깨짐 방지
        handler = logging.FileHandler(log_filename, encoding='utf-8-sig')
        formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
        # 콘솔에도 실시간 출력 추가 (선택 사항)
        console = logging.StreamHandler()
        console.setFormatter(formatter)
        logger.addHandler(console)
        
    return logger

# 3. 직렬 실행 엔진
def run_serial_group(group_name, script_paths, success_flag):
    logger = setup_logger(group_name)
    logger.info(f"=== {group_name} 파이프라인 시작 ===")
    
    for script in script_paths:
        full_path = os.path.join(BASE_DIR, script)
        if not os.path.exists(full_path):
            logger.error(f"파일을 찾을 수 없음: {full_path}")
            success_flag.value = 0
            return
            
        logger.info(f"▶ 실행 중: {script}")
        try:
            # 파이썬 3.10 윈도우 인코딩 에러(UnicodeDecodeError) 방지
            result = subprocess.run(
                [sys.executable, full_path],
                check=True,
                capture_output=True,
                text=True,
                errors='replace' # 깨진 문자는 대체 문자로 치환하여 중단 방지
            )
            logger.info(f"✔ 성공: {script}")
            # 스크립트 내부의 print문 결과도 로그에 기록하고 싶다면 아래 주석 해제
            # logger.info(f"출력 결과:\n{result.stdout}")
            
        except subprocess.CalledProcessError as e:
            logger.error(f"❌ 에러 발생: {script}")
            logger.error(f"Exit Code: {e.returncode}")
            logger.error(f"상세 에러 내용:\n{e.stderr}")
            success_flag.value = 0
            return 
            
    logger.info(f"=== {group_name} 모든 단계 완료 ===")
    success_flag.value = 1

# 4. 메인 제어 로직
if __name__ == "__main__":
    # 공유 메모리 설정 (성공 여부 체크)
    success_sca = Value('i', 1)
    success_stress = Value('i', 1)

    # --- PHASE 1: 알고리즘 학습 ---
    sca_scripts = [
        r"Early_Guard\SCA_signal_preprocessing_final.py",
        r"Early_Guard\normal_signal_preprocessing_final.py",
        r"Early_Guard\run_early.py"
    ]
    
    stress_scripts = [
        r"Condi_Guard\src\ecg_preprocessing.py",
        r"Condi_Guard\src\ppg_preprocessing.py",
        r"Condi_Guard\src\run_condi.py"
    ]

    print(f"\n📂 로그 저장 폴더: {LOG_DIR}")
    print("🚀 [Phase 1] 심정지 및 스트레스 알고리즘 학습 시작 (병렬)...")
    
    p1 = Process(target=run_serial_group, args=("SCA_Algorithm", sca_scripts, success_sca))
    p2 = Process(target=run_serial_group, args=("Stress_Algorithm", stress_scripts, success_stress))

    p1.start(); p2.start()
    p1.join(); p2.join()

    # 에러 체크
    if success_sca.value == 0 or success_stress.value == 0:
        print("\n❌ Phase 1 중 오류가 발생하여 공정을 중단합니다. logs 폴더를 확인하세요.")
        sys.exit(1)

    print("\n" + "="*50)
    print("✨ 모든 분석 파이프라인 공정이 성공적으로 종료되었습니다.")
    print("="*50)