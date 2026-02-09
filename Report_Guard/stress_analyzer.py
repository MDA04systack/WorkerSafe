import os
import sys
import numpy as np
import pandas as pd
import shap
import sklearn.pipeline
import config  #

# 1. DisCERN 경로 자동 등록
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
#
discern_path = os.path.join(project_root, "Condi_Guard", "src", "xai", "discern")

if discern_path not in sys.path:
    sys.path.insert(0, discern_path)

from discern_tabular import DisCERNTabular  #

# 2. 회귀 점수(NasaTLX)를 상태 라벨(0, 1, 2)로 변환하는 함수
def get_condi_label(score):
    if score >= config.CONDI_THRESHOLDS["HIGH_LIMIT"]:
        return 2  # High Load (위험)
    elif score >= config.CONDI_THRESHOLDS["LOW_LIMIT"]:
        return 1  # Medium Load (보통)
    else:
        return 0  # Low Load (쾌적)

# 3. 회귀 모델을 분류 모델처럼 동작하게 만드는 래퍼 클래스
class ModelWrapper:
    def __init__(self, original_model):
        self.model = original_model
        
    def predict(self, x):
        # 모델의 예측 점수를 가져와서 라벨(0, 1, 2)로 즉시 변환합니다.
        scores = self.model.predict(x)
        return np.array([get_condi_label(s) for s in scores])

def run_stress_xai(model, sample_input, train_df, feature_names):
    """
    NasaTLX 회귀 모델을 기반으로 SHAP 중요도와 DisCERN 개선안을 계산합니다.
    """
    
    # [1] 모델 래핑 (회귀 -> 분류 변환)
    wrapped_model = ModelWrapper(model)
    
    # [2] 현재 데이터의 예측 점수 및 라벨 계산
    current_score = model.predict(sample_input.reshape(1, -1))[0]
    current_label = get_condi_label(current_score)
    
    # [3] 배경 데이터의 예측 라벨 생성 (DisCERN 학습용)
    # 실제 정답(condition)이 아닌 모델의 예측 라벨을 사용해야 정확합니다.
    train_data_values = train_df[feature_names].values
    train_labels_pred = wrapped_model.predict(train_data_values)

    # [4] DisCERN 엔진 초기화 및 개선안 탐색
    # threshold를 살짝 높여(0.05) 개선안을 더 잘 찾도록 설정합니다.
    engine = DisCERNTabular(model=wrapped_model, attrib='SHAP', threshold=0.05)
    engine.init_data(
        train_data=train_data_values, 
        train_labels=train_labels_pred, 
        feature_names=feature_names,
        labels=[0, 1, 2],
        cat_feature_indices=[],
        immutable_feature_indices=[]
    )
    
    try:
        # 현재 상태(current_label)에서 상태가 변하는 지점 탐색
        cf_norm, cf_label, sparsity, proximity = engine.find_cf(sample_input, current_label)
    except Exception as e:
        print(f"⚠️ CF 탐색 실패: {e}")
        cf_norm, cf_label = sample_input, current_label

    # [5] SHAP 계산 (파이프라인 대응)
    # 파이프라인 내부의 실제 모델(트리 등)을 추출합니다.
    final_model = model.steps[-1][1] if hasattr(model, 'steps') else model
    
    try:
        explainer = shap.TreeExplainer(final_model)
        # 회귀 모델이므로 단일 배열이 반환됩니다.
        shap_values = explainer.shap_values(sample_input.reshape(1, -1))[0]
    except Exception:
        # 트리 모델이 아닐 경우 범용 Explainer 사용
        explainer = shap.Explainer(model.predict, train_df[feature_names])
        shap_values = explainer(sample_input.reshape(1, -1)).values[0]

    # [6] Report_Guard_Condi.py의 7개 규격에 맞춰 결과 반환
    return [
        sample_input.flatten(),   # res[0]: 현재_norm
        cf_norm.flatten(),        # res[1]: 추천_norm
        current_score,            # res[2]: 현재 NasaTLX 점수
        cf_label,                 # res[3]: 목표 상태(0, 1, 2)
        sample_input.flatten(),   # res[4]: 현재수치
        cf_norm.flatten(),        # res[5]: 목표수치
        shap_values               # res[6]: SHAP 중요도
    ]