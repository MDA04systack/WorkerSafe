### 개인별 맞춤형 모델

import os
import numpy as np
import pandas as pd
import joblib

#from src import utils, paths
from sklearn.model_selection import cross_validate
import paths
import utils


# ------------------------------------------------------ 
# 함수: get_cross_val_results
# 목적: 주어진 모델과 데이터에 대해 10-Fold 교차 검증을 수행하고 지표를 반환함 
# ------------------------------------------------------ 
def get_cross_val_results(model_type, clf, X, y):
    utils.validate_model_type_name(model_type=model_type)
    # 평가지표 설정: 분류(정확도 등) 또는 회귀(오차 등)에 따라 다름
    if model_type == "classification":

        scoring = ["accuracy", "balanced_accuracy", "f1_micro", "f1_macro", "precision_micro",
                   "recall_micro", "precision_macro", "recall_macro"]
    else: # model_type이 regression 일 경우
        scoring = ["explained_variance", "max_error", "neg_mean_absolute_error", "neg_mean_squared_error",
                   "neg_mean_squared_log_error", "neg_median_absolute_error"]
    
    # 교차 검증 수행: 데이터를 10개로 나누어 학습 및 평가 반복
    scores = cross_validate(clf, X=X, y=y, scoring=scoring, cv=10)
    result = pd.DataFrame.from_dict(scores, orient='columns')
    return result

# ------------------------------------------------------ 
# 함수: generate_person_specific_results
# 목적: 피험자별 데이터를 독립적으로 학습시켜 개인화된 모델을 생성하고 저장함 
# ------------------------------------------------------ 
def generate_person_specific_results(dataset, signal, model_type):
    # 1. 데이터 로드 확인
    data = utils.get_combined_data(dataset=dataset, signal=signal)
    
    # [디버깅 추가] 데이터가 제대로 로드되었는지 확인
    print(f"\n>>> [디버깅] {dataset} - {signal} - {model_type} 시작")
    print(f">>> 데이터 로드 성공: 행 수 = {len(data)}")

    subject_id_col = utils.get_subject_id_column(dataset)
    subjects_ids = sorted(data[subject_id_col].unique())
    print(f">>> 감지된 피험자 리스트: {subjects_ids}")

    # 2. 특징(Feature) 목록 확인
    target = utils.get_prediction_target(dataset=dataset, model_type=model_type)
    features = utils.get_important_features(dataset=dataset, signal=signal, model_type=model_type)

    # [디버깅 추가] 특징 목록이 비어있는지 확인 (가장 의심되는 지점)
    print(f">>> 추출된 특징 목록 ({len(features)}개): {features}")
    if len(features) == 0:
        print(f"⚠️ 경고: {dataset}-{signal}에 대한 특징(Feature) 목록이 비어 있습니다!")
        return # 더 이상 진행하지 않음

    # 3. 모델 루프 시작

    models = utils.get_model(model_type)
    for clf in models:
        model_name = type(clf).__name__
        out_dir = paths.ensure_directory_exists(os.path.join(paths.result_directory(), "model-performance",
                                                             model_type, dataset, signal, "person-specific",
                                                             model_name))
        for subject_id in subjects_ids:
            # 개인 데이터 추출: 해당 피험자의 데이터만 분리
            df = data.loc[data[subject_id_col] == subject_id]
            X, y = df[features], df[target]
            
            # 파이프라인 생성 및 개별 학습: 교차검증과 별개로 전체 개인 데이터로 학습
            clf_pipe = utils.get_pipeline_model(clf)
            clf_pipe.fit(X, y)
            
            # 모델 저장: 개인 전용 모델 파일 저장
            joblib.dump(clf_pipe, os.path.join(out_dir, f"person_specific_model_{int(subject_id)}.pkl"))
            
            # 결과 저장: 교차 검증 성능 기록
            result = get_cross_val_results(model_type, clf_pipe, X, y)
            result.to_csv(os.path.join(out_dir, "subject_" + str(int(subject_id)) + ".csv"), index=True)