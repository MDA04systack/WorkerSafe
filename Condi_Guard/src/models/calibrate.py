## 하이브리드 보정 모델 생성 및 학습 코드: 범용 데이터에 소량의 개인 데이터를 추가하여 모델을 미세 조정(Fine-tuning)하는 코드
import os
import pandas as pd
import joblib
from sklearn.utils import shuffle

# [수정] INT 루트를 기준으로 절대 경로 임포트 수행
import paths 
import utils

# ------------------------------------------------------ 
# 함수: calibrate_model
# 목적: 범용 데이터와 개인 보정 데이터를 합쳐서 새로운 모델을 학습시킴 
# ------------------------------------------------------ 
def calibrate_model(model, generic_samples, calibration_samples, target, features):
    from sklearn.base import clone
    # 데이터 통합: 범용 데이터셋과 소량의 개인 데이터 결합
    train = pd.concat([generic_samples, calibration_samples])
    train = shuffle(train)
    X_train = train[features]
    y_train = train[target]
    
    # 모델 초기화: 기존 학습 상태를 지우고 새로 학습하기 위해 복제
    clf = clone(model)
    clf.fit(X_train, y_train)
    return clf

# ------------------------------------------------------ 
# 함수: 
# 목적: 
# ------------------------------------------------------ 
def get_data(dataset, signal):
    root_dir = os.path.join(paths.data_directory(), signal, dataset)
    calibration = pd.read_csv(os.path.join(root_dir, "calibration.csv"))
    generic = pd.read_csv(os.path.join(root_dir, "generic.csv"))
    test = pd.read_csv(os.path.join(root_dir, "test.csv"))
    return generic, calibration, test

# ------------------------------------------------------ 
# 함수: generate_calibration_results
# 목적: 다양한 보정 데이터 크기에 대해 모델을 보정하고 성능을 측정 및 저장함 
# ------------------------------------------------------ 
def generate_calibration_results(dataset, signal, model_type):
    target = utils.get_prediction_target(dataset=dataset, model_type=model_type)
    # 데이터 로드: 범용, 보정용, 테스트용 데이터셋 각각 로드
    generic_data, calibration_data, test = get_data(dataset, signal)
    features = utils.get_important_features(dataset=dataset, signal=signal, model_type=model_type)
    X_test = test[features]
    y_test = test[target]
    
    # 보정 샘플 크기 설정: 몇 개의 데이터를 보정에 사용할지 결정
    calibration_sample_size =utils. get_calibration_sample_sizes(dataset=dataset)
    models =utils. get_model(model_type)
    for clf in models:
        model_name = type(clf).__name__
        out_dir = paths.ensure_directory_exists(os.path.join(paths.result_directory(), "model-performance",
                                                             model_type, dataset, signal, "calibration", model_name))

        for size in calibration_sample_size:
            # 샘플 추출: 보정용 데이터에서 정해진 크기만큼 무작위 추출
            calibration_samples = calibration_data.sample(n=size, random_state=0)
            clf_pipe = utils.get_pipeline_model(clf)
            
            # 보정 수행: 범용 데이터 + 추출된 개인 샘플로 재학습
            calibrated_clf = calibrate_model(model=clf_pipe, generic_samples=generic_data,
                                             calibration_samples=calibration_samples, target=target,
                                             features=features)

            # 모델 저장: 보정된 크기 정보를 포함하여 저장
            joblib.dump(calibrated_clf, os.path.join(out_dir, f"calibrated_model_size_{size}.pkl"))

            # 검증: 보정된 모델의 테스트 데이터 성능 기록
            predictions = calibrated_clf.predict(X_test)
            result = utils.get_prediction_metrics(model_type, predictions=predictions, y_test=y_test)
            result.to_csv(os.path.join(out_dir, str(int(size / 4)) + ".csv"))




