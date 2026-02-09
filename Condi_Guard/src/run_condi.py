# Condi Guard 메인 실행 코드

import sys
import os
import warnings
import itertools # 다중 루프를 효율적으로 처리하기 위한 표준 라이브러리

# 1. 현재 파일의 위치를 기준으로 프로젝트 루트(INT)를 정확히 계산
current_dir = os.path.dirname(os.path.abspath(__file__)) # D:\Semi2\INT\Condi_Guard\src
project_root = os.path.abspath(os.path.join(current_dir, "..", "..")) # D:\Semi2\INT

# 2. 시스템 경로에 루트 추가 (모든 모듈 임포트의 기준점)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 3. 루트에 있는 공통 모듈 임포트
import paths 

# 4. Report_Guard 패키지 내의 config 임포트
# [중요] 반드시 Report_Guard 폴더 내에 __init__.py 파일이 있어야 합니다.
from Report_Guard import config 

# 5. 자기 자신의 패키지 경로를 사용하여 모듈 임포트
from Condi_Guard.src.models import calibrate, generic, person_specific_model
from Condi_Guard.src.analysis import tables, plots

# 2. 실험 환경 설정 (Centralized Configuration)
# 실험에 사용할 변수들을 이곳에서 한 번에 관리합니다. 
# 새로운 데이터셋이나 신호를 추가할 때 이 딕셔너리만 수정하면 됩니다.
CONFIG = {
    "datasets": ["swell"],           # 대상 데이터셋 (swell, wesad 등)
    "signals": ["hrv"],              # 분석 신호 (hrv, eda 등)
    "model_types": ["regression"],   # 분석 유형 (classification, regression)
    "comparison_models": ["RandomForest"], # 비교 실험용 기본 모델
    "calibration_models": ["ExtraTrees"]   # 하이브리드 보정용 정밀 모델
}

# 3. 핵심 실행 엔진: task_runner (Execution Engine)
# 이 함수는 기존 코드에서 반복되던 3중 for 루프를 대신 수행합니다.
# '작업 함수(func)'를 인자로 받아 설정된 모든 조합(dataset x signal x model_type)에 대해 실행합니다.
def task_runner(func, use_model_type_loop=True, **kwargs):
    """
    func: 실행할 함수 (예: calibrate.generate_calibration_results)
    use_model_type_loop: 
        - True: model_type(분류/회귀)별로 함수를 각각 실행 (일반적인 경우)
        - False: model_types 리스트 전체를 함수에 한 번에 전달 (시각화 함수용)
    **kwargs: 함수에 전달할 추가 인자 (예: model_name="RandomForest")
    """
    
    # itertools.product는 중첩된 루프를 하나로 합쳐줍니다.
    # 예: [swell] x [hrv] -> (swell, hrv) 조합 생성
    combinations = itertools.product(CONFIG["datasets"], CONFIG["signals"])
    
    for dataset, signal in combinations:
        if use_model_type_loop:
            # Case A: 각 분석 유형(분류/회귀)마다 함수를 따로 호출해야 하는 경우
            for m_type in CONFIG["model_types"]:
                # f-string의 :.<40은 로그 정렬을 위해 사용합니다.
                print(f"[실행 중] {func.__name__:.<40} | {dataset} | {signal} | {m_type}")
                # 실제 함수 호출 (예: tables.generate_generic_result_table(dataset=..., ...))
                func(dataset=dataset, signal=signal, model_type=m_type, **kwargs)
        else:
            # Case B: 시각화(Plot)처럼 여러 model_type을 한 그래프에 그릴 때 (리스트 형태로 전달)
            print(f"[실행 중] {func.__name__:.<40} | {dataset} | {signal} | 리스트 전체 전달")
            func(dataset=dataset, signal=signal, model_types=CONFIG["model_types"], **kwargs)

# 4. 메인 실행 흐름 제어
def main():
    # 불필요한 경고 메시지 출력 방지
    warnings.simplefilter(action='ignore')
    
    print("\n" + "="*80)
    print("      Condi Guard: 통합 분석 파이프라인 가동")
    print("="*80 + "\n")

    # --- [STEP 1] 모델 학습 및 원천 데이터 생성 ---
    # 데이터 생성은 시간이 오래 걸리므로, 필요할 때만 주석을 해제하여 실행하세요.
    #task_runner(person_specific_model.generate_person_specific_results) # 개인별 모델 학습
    #task_runner(calibrate.generate_calibration_results)              # 하이브리드 보정 수행
    #task_runner(generic.generate_generic_results)                  # 일반 모델(LOSO-CV) 학습
    #task_runner(generic.generate_combined_general_results)         # 전체 통합 데이터 학습

    # --- [STEP 2] 결과 분석 표(Table) 생성 ---
    # 각 실험 결과 파일들을 취합하여 CSV나 Excel 형태의 리포트를 생성합니다.
    #task_runner(tables.generate_person_specific_result_table)
    #task_runner(tables.generate_generic_result_table)
    #task_runner(tables.generate_calibration_table)

    # --- [STEP 3] 결과 시각화 (Plots) ---
    # 학습된 결과 데이터를 읽어와서 논문에 들어갈 그래프(Fig)를 생성합니다.
    
    # 3-1. 일반 모델 vs 개인형 모델 비교 (RandomForest 기준)
    for model in CONFIG["comparison_models"]:
        task_runner(plots.plot_generic_vs_person_specific_model, 
                    use_model_type_loop=False, 
                    model_name=model)
    
    # 3-2. 보정 샘플 수 증가에 따른 성능 향상 곡선 (ExtraTrees 기준)
    for model in CONFIG["calibration_models"]:
        task_runner(plots.plot_calibration_result, 
                    use_model_type_loop=False, 
                    model_name=model)

    print("\n" + "="*80)
    print("      모든 분석 작업이 완료되었습니다. (결과물: /results 폴더 확인)")
    print("="*80 + "\n")

if __name__ == "__main__":
    main()