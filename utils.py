### Workersafe 유틸리티/기능 담당 코드 파일: 데이터 로드, 성능 지표 계산(MAE, RMSE, Accuracy), 머신러닝 모델 생성(ExtraTrees, RandomForest), 특징 추출 등

# 데이터 로드, 모델 설정, 성능 지표 계산 및 데이터 검증을 담당하는 핵심 유틸리티 함수들로 구성됨

import numpy as np
import pandas as pd
import os
from sklearn import metrics

# 상대 임포트(from . import paths) 대신 직접 임포트
import paths

# --- 성능 지표 계산 관련 함수 ---

# 회귀 모델 성능 평가: MAE(평균 절대 오차) 및 RMSE(평균 제곱근 오차) 산출
def get_regression_metrics(predictions, y_test):
    mae = metrics.mean_absolute_error(y_test, predictions) # 평균 절대 오차
    rmse = np.sqrt(metrics.mean_squared_error(y_test, predictions)) # 평균 제곱근 오차
    return pd.DataFrame({'MAE': mae, 'RMSE': rmse}, index=[0]).transpose()

# 분류 모델 성능 평가: 불균형 데이터를 고려한 가중 평균(Weighted) 지표 산출
def get_classification_metrics(predictions, y_test):
    accuracy = metrics.accuracy_score(y_pred=predictions, y_true=y_test)
    # weighted average를 사용하여 불균형 클래스 문제 대응
    precision = metrics.precision_score(y_pred=predictions, y_true=y_test, average='weighted')
    f1_score = metrics.f1_score(y_pred=predictions, y_true=y_test, average='weighted')
    recall = metrics.recall_score(y_pred=predictions, y_true=y_test, average='weighted')
    result = {'Accuracy': accuracy, 'Precision': precision, 'Recall': recall,
              "F1 score": f1_score}
    return pd.DataFrame(result, index=[0]).transpose()

# --- 특징(Feature) 및 데이터 로드 관련 함수 ---

# 데이터셋/신호별 중요 특징 추출: MDI 기반 중요도 상위 특징 선별 (40~75개)
def get_important_features(dataset, signal, model_type):
    in_file = os.path.join(paths.result_directory(), "feature-ranks", signal,
                           model_type, dataset, "features-ranks.csv")
    
    # 하드코딩된 수치를 매핑 테이블로 관리하여 유지보수 용이성 확보
    feature_counts = {
        ("swell", "eda"): 46,
        ("swell", "hrv"): 75,
        ("wesad", "hrv"): 40,
        ("wesad", "eda"): 45
    }
    
    feature_count = feature_counts.get((dataset, signal), 0)
    
    if not os.path.exists(in_file):
        raise FileNotFoundError(f"중요 특징 파일이 없습니다: {in_file}")
        
    data = pd.read_csv(in_file)
    features = data.iloc[:, 0].head(feature_count).tolist() # list(data)[0] 대신 iloc 사용 권장
    return features

# 학습 파이프라인 구축: RobustScaler 적용으로 생체 신호 이상치 및 개인 편차 보정
def get_pipeline_model(model):
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import RobustScaler
    pipeline = Pipeline([('scaler', RobustScaler()), ('model', model)])
    return pipeline

# 분류 모델 설정: 안정성용 RF(Depth 2) 및 패턴 학습용 ExtraTrees(Depth 16) 생성
def get_classifier():
    from sklearn.ensemble import ExtraTreesClassifier
    from sklearn.ensemble import RandomForestClassifier
    
    xtree = ExtraTreesClassifier(n_estimators=1000, class_weight="balanced", max_depth=16,
                                 oob_score=True, bootstrap=True, n_jobs=-1, max_features='sqrt', random_state=0)
    rf = RandomForestClassifier(n_estimators=1000, oob_score=True, max_features='sqrt',
                                bootstrap=True, max_depth=2, class_weight="balanced", random_state=0, n_jobs=-1)
    models = [xtree, rf]
    return models

# 회귀 모델 설정: 스트레스 점수 예측을 위한 Regressor 세부 설정
def get_regressor():
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.ensemble import ExtraTreesRegressor
    
    # max_features='auto'를 1.0 또는 'sqrt'로 변경
    xtree = ExtraTreesRegressor(n_estimators=1000, max_depth=16, oob_score=True, bootstrap=True,
                                max_features=1.0, random_state=0, n_jobs=-1) # 'auto' -> 1.0
    rf = RandomForestRegressor(n_estimators=1000, max_features=(1 / 3), max_depth=2,
                               random_state=0, oob_score=True, bootstrap=True, n_jobs=-1)
    models = [xtree, rf]
    return models

# 실험 대상 피험자 필터링: 유효 데이터 보유 피험자 ID 리스트 반환
def get_subjects_ids(dataset):
    validate_dataset_name(dataset)
    if dataset == "swell": # SWELL: 25명 중 20명 선별
        return [x for x in range(1, 26) if x not in [8, 11, 14, 15, 23]]
    else: # WESAD: 15명 중 13명 선별
        return [x for x in range(2, 18) if x not in [5, 12]]

# 모델 타입에 따른 성능 지표 선택 호출
def get_prediction_metrics(model_type, predictions, y_test):
    validate_model_type_name(model_type=model_type)
    if model_type == "classification":
        return get_classification_metrics(predictions=predictions, y_test=y_test)
    else:
        return get_regression_metrics(predictions=predictions, y_test=y_test)

# 데이터셋별 예측 타겟 설정: 분류(Condition) 및 회귀(NasaTLX/SSSQ 점수) 구분
def get_prediction_target(dataset, model_type):
    validate_dataset_name(dataset)
    validate_model_type_name(model_type)
    target = None
    if model_type == "classification":
        target = 'condition'
    elif dataset == "swell" and model_type == "regression":
        target = 'NasaTLX'
    elif dataset == "wesad" and model_type == "regression":
        target = "SSSQ"
    return target

# 통합 데이터 로드: 전처리 완료된 전체 샘플 파일 로드
def get_combined_data(dataset, signal):
    in_file = os.path.join(paths.data_directory(), signal, dataset, "all-samples.csv")
    return pd.read_csv(in_file)

# 모델 타입별 알고리즘 리스트 반환
def get_model(model_type):
    if model_type == "classification":
        clf = get_classifier()
    else:
        clf = get_regressor()
    return clf

# 인자 유효성 검사 함수 (Model Type)
def validate_model_type_name(model_type):
    if model_type not in ["classification", "regression"]:
        raise ValueError("{0} is an invalid model type".format(model_type))

# 인자 유효성 검사 함수 (Dataset)
def validate_dataset_name(dataset):
    if dataset not in ["wesad", "swell"]:
        raise ValueError("{0} is an invalid dataset name".format(dataset))

# 인자 유효성 검사 함수 (Signal)
def validate_signal_name(signal):
    if signal not in ["eda", "hrv"]:
        raise ValueError(f"{signal} is an invalid signal name")


if __name__ == '__main__':
    f1 = get_important_features(dataset="swell", signal="eda", model_type="regression")
    f2 = get_important_features(dataset="swell", signal="eda", model_type="classification")
    f3 = get_important_features(dataset="swell", signal="hrv", model_type="regression")
    f4 = get_important_features(dataset="swell", signal="hrv", model_type="classification")
    f5 = get_important_features(dataset="wesad", signal="hrv", model_type="classification")
    f6 = get_important_features(dataset="wesad", signal="hrv", model_type="regression")
    f7 = get_important_features(dataset="wesad", signal="eda", model_type="regression")
    f8 = get_important_features(dataset="wesad", signal="eda", model_type="classification")

# 데이터셋별 피험자 ID 컬럼 명칭 반환
def get_subject_id_column(dataset):
    if dataset == "swell":
        subject_id_col = "subject_id"
    else:
        subject_id_col = "subject id"
    return subject_id_col

# 하이브리드 보정용 샘플 사이즈 구간 설정 (최대 400개)
def get_calibration_sample_sizes(dataset):
    validate_dataset_name(dataset=dataset)
    # SWELL: 40개 단위, WESAD: 20개 단위로 증분 설정
    if dataset == "swell":
        return np.arange(0, 401, 40)
    else:
        return np.arange(0, 401, 20)
