## 일반 모델 생성 및 학습 코드: 여러 사용자의 데이터를 통합하여 보편적인 스트레스 패턴을 학습
import os
import joblib # 모델 객체 저장을 위한 라이브러리
from sklearn.utils import shuffle
import paths
import utils
import pandas as pd

# ------------------------------------------------------ 
# 함수: generate_generic_results
# 목적: Leave-One-Subject-Out 방식으로 범용 모델의 성능을 측정하고 모델을 저장함
# ------------------------------------------------------ 
def generate_generic_results(dataset, signal, model_type):
    # 데이터 로드: 지정된 데이터셋과 신호(HRV/EDA) 데이터를 결합하여 가져옴
    data = utils.get_combined_data(dataset=dataset, signal=signal)
    # 피험자 ID 컬럼 확인: 데이터셋별 피험자 식별자 추출
    subject_id_col = utils.get_subject_id_column(dataset)
    subjects_ids = sorted(data[subject_id_col].unique())
    # 타겟 및 피처 추출: 예측 목표(스트레스 점수 등)와 중요 피처 목록 확보
    target = utils.get_prediction_target(dataset=dataset, model_type=model_type)
    features = utils.get_important_features(dataset=dataset, signal=signal, model_type=model_type)
    
    # 모델 리스트 가져오기: 설정된 모델 타입(분류/회귀)에 따른 알고리즘들
    models = utils.get_model(model_type)
    for clf in models:
        model_name = type(clf).__name__
        # 출력 디렉토리 생성: 결과 및 모델을 저장할 경로 설정
        out_dir = paths.ensure_directory_exists(os.path.join(paths.result_directory(), "model-performance",
                                                             model_type, dataset, signal, "generic-model",
                                                             model_name))
        for subject_id in subjects_ids:
            # 파이프라인 구성: 전처리(Scaler)와 모델이 포함된 객체 생성
            clf = utils.get_pipeline_model(clf)
            # 데이터 분할: 특정 피험자 1명을 Test로, 나머지를 Train으로 설정
            train_subjects = [x for x in subjects_ids if x != subject_id]
            test = data.loc[data[subject_id_col] == subject_id]
            train = data.loc[data[subject_id_col].isin(train_subjects)]
            
            # 데이터 셔플: 학습 데이터의 순서를 무작위로 섞음
            train = shuffle(train)
            test = shuffle(test)
            X_train = train[features]
            y_train = train[target]
            X_test = test[features]
            y_test = test[target]
            
            # 모델 학습: 범용 데이터로 학습 수행
            clf.fit(X_train, y_train)
            
            # 모델 저장: 학습된 파이프라인(Scaler+Model)을 .pkl 파일로 저장
            joblib.dump(clf, os.path.join(out_dir, f"generic_model_test_subj_{int(subject_id)}.pkl"))          
            
            # 예측 및 성능 기록: 테스트 데이터를 통한 성능 검증
            predictions = clf.predict(X_test)
            result = utils.get_prediction_metrics(model_type, predictions=predictions, y_test=y_test)
            print("user_id {0} \t{1}".format(subject_id, result.transpose()))

            result.to_csv(os.path.join(out_dir, "subject_" + str(int(subject_id)) + ".csv"), index=True)

# ------------------------------------------------------ 
# 함수: generate_combined_general_results
# 목적: 데이터셋 전체를 하나로 통합하여 모델의 전반적인 교차 검증 성능을 측정함
# ------------------------------------------------------ 
def generate_combined_general_results(dataset, signal, model_type):
    # 데이터 통합: 모든 피험자의 데이터를 구분 없이 하나로 합침
    data = utils.get_combined_data(dataset=dataset, signal=signal)
    
    # 목표 및 피처 설정: 예측 대상과 모델에 입력될 특징 추출
    target = utils.get_prediction_target(dataset=dataset, model_type=model_type)
    features = utils.get_important_features(dataset=dataset, signal=signal, model_type=model_type)
    
    # 모델 루프: 설정된 모든 알고리즘에 대해 성능 평가 수행
    models = utils.get_model(model_type)
    for clf in models:
        model_name = type(clf).__name__
        # 경로 설정: 'combined-general-model' 폴더 아래 결과 저장 경로 확보
        out_dir = paths.ensure_directory_exists(os.path.join(paths.result_directory(), "model-performance",
                                                             model_type, dataset, signal, "combined-general-model",
                                                             model_name))
        
        # 파이프라인 적용: 모델에 필요한 전처리(Scaler 등) 연결
        clf = utils.get_pipeline_model(clf)
        X = data[features]
        y = data[target]
        
        # 교차 검증 실행: 하단의 get_cross_val_results 함수 호출
        result = get_cross_val_results(model_type, clf, X, y)
        
        # [추가 제안] 전체 데이터를 학습시킨 최종 범용 모델 저장
        # 이 모델은 특정 개인에게 치우치지 않은 '표준 스트레스 예측기' 역할을 함
        clf.fit(X, y) 
        joblib.dump(clf, os.path.join(out_dir, f"final_combined_general_{model_name}.pkl"))
        
        # 결과 저장: 각 Fold별 성능 지표를 CSV로 기록
        result.index.name = 'CV Fold'
        result.to_csv(os.path.join(out_dir, model_name + ".csv"), index=True)

# ------------------------------------------------------ 
# 함수: get_cross_val_results
# 목적: 모델의 신뢰성을 높이기 위해 데이터를 10개 그룹으로 나누어 반복 검증(10-Fold CV)함
# ------------------------------------------------------ 
def get_cross_val_results(model_type, clf, X, y):
    from sklearn.model_selection import cross_validate
    utils.validate_model_type_name(model_type=model_type)
    # 지표 설정 (Classification): 분류 모델일 때 정확도, 정밀도, 재현율 등 측정
    if model_type == "classification":

        scoring = ["accuracy", "balanced_accuracy", "f1_micro", "f1_macro", "precision_micro",
                   "recall_micro", "precision_macro", "recall_macro"]
    # 지표 설정 (Regression): 회귀 모델일 때 절대 오차(MAE), 제곱 오차(MSE) 등 측정
    else:
        scoring = ["explained_variance", "max_error", "neg_mean_absolute_error", "neg_mean_squared_error",
                   "neg_mean_squared_log_error", "neg_median_absolute_error"]
    scores = cross_validate(clf, X=X, y=y, scoring=scoring, cv=10)
    result = pd.DataFrame.from_dict(scores, orient='columns')
    return result