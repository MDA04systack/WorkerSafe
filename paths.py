### Workersafe 경로 관리 담당 코드: 프로젝트 내의 데이터, 모델, 결과 저장소 등 모든 디렉토리의 절대 경로를 관리

import os
import sys

# 1. 프로젝트 최상위 루트(INT) 경로 계산
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 2. 스트레스 분석(Condi Guard) 소스 및 시스템 경로 등록
SRC_PATH = os.path.join(BASE_DIR, "Condi_Guard", "src")
for p in [BASE_DIR, SRC_PATH]:
    if p not in sys.path:
        sys.path.insert(0, p)

def ensure_directory_exists(folder):
    """폴더가 없으면 생성하는 안전 함수"""
    if not os.path.exists(folder):
        os.makedirs(folder, exist_ok=True)
    return folder

# --- 3. Condi Guard (스트레스) 관련 디렉토리 경로 ---

def data_directory():
    """스트레스 데이터셋(calibration.csv 등) 위치: Condi_Guard/dataset"""
    p = os.path.join(BASE_DIR, "Condi_Guard", "dataset")
    return ensure_directory_exists(os.path.normpath(p))

def result_directory():
    """스트레스 분석 결과 저장 위치: Condi_Guard/src/results"""
    p = os.path.join(SRC_PATH, "results")
    return ensure_directory_exists(os.path.normpath(p))

def plots_directory():
    """시각화 결과 저장 폴더 경로 반환"""
    path = os.path.join(BASE_DIR, "results")
    return ensure_directory_exists(path)

# --- 4. 모델 경로 추출 및 상수 정의 ---

def get_model_path(model_name="ExtraTreesRegressor", method="calibration", size=400):
    """
    스트레스 예측 모델의 세분화된 폴더 구조를 찾아 경로를 반환합니다.
    구조: results/model-performance/regression/swell/hrv/{method}/{model_name}/...
    """
    p = os.path.join(
        result_directory(),
        "model-performance",
        "regression",
        "swell",
        "hrv",
        method,
        model_name,
        f"calibrated_model_size_{size}.pkl"
    )
    return os.path.normpath(p)

# 상위 모듈(config.py 등)에서 참조하는 경로 상수
CALIBRATED_MODEL_PATH = get_model_path(model_name="ExtraTreesRegressor", method="calibration", size=400)
CALIBRATION_DATA_PATH = os.path.join(data_directory(), "hrv", "swell", "calibration.csv")


# --- 5. Early Guard (심정지 위험 SCA) 관련 경로 ---

def sca_dir():
    """심정지 예측 폴더(Early_Guard) 경로 반환"""
    return os.path.normpath(os.path.join(BASE_DIR, "Early_Guard"))

def sca_model_path():
    """SCA 딥러닝 모델 파일(.keras)의 절대 경로 반환"""
    return os.path.join(sca_dir(), "saved_models", "SCA_1D_CNN_Model.keras")

def sca_dataset_dir():
    """SCA 관련 데이터셋(.npy, .csv)이 있는 폴더 경로 반환"""
    return os.path.join(sca_dir(), "dataset")

# 상세 데이터 파일 경로 추가 (유연성 확보)
def sca_x_test_path():
    return os.path.join(sca_dataset_dir(), "X_test_final.npy")

def sca_metadata_path():
    return os.path.join(sca_dataset_dir(), "test_metadata_check.csv")

def report_output_directory():
    """생성된 PDF 리포트 저장 폴더"""
    p = os.path.join(BASE_DIR, "Report_Guard", "outputs")
    return ensure_directory_exists(os.path.normpath(p))