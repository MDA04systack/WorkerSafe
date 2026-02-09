# ---------------------------------------------------------
# 생체 신호 분석 결과 수치 데이터(Table) 생성 모듈
# ---------------------------------------------------------
import pandas as pd
import numpy as np
import paths
import utils
import os

# ---------------------------------------------------------
# 1. 개인별 모델 결과 표 생성 (Person-Specific Table)
# ---------------------------------------------------------
def generate_person_specific_result_table(dataset, signal, model_type):
    # # 대상 피험자 리스트 및 출력용 한글 컬럼명 설정
    subject_ids = utils.get_subjects_ids(dataset=dataset)
    subjects_kor = [f"피험자_{x}" for x in subject_ids]
    # # 교차 검증 폴드(10-Fold) 규격 정의
    n_folds = 10
    out_df = pd.DataFrame(columns=subjects_kor, index=np.arange(n_folds))
    
    # # 분석 모델 리스트 확보
    models = utils.get_model(model_type=model_type)
    
    for clf in models:
        model_name = type(clf).__name__
        # 경로 설정 시 os.path.join을 사용하여 슬래시 중복 방지
        in_dir = os.path.join(paths.result_directory(), "model-performance", model_type,
                              dataset, signal, "person-specific", model_name)
        
        # 출력 파일명 정의
        file_name = f"{dataset}_{signal}_{model_type}_개인별_결과.xlsx"
        out_dir = paths.ensure_directory_exists(os.path.join(paths.result_directory(), "tables", model_type,
                                                         dataset, signal, "person-specific", model_name))
        out_path = os.path.join(out_dir, file_name)

        # 데이터 존재 확인 후 엑셀 작성 시작
        if not os.path.exists(os.path.join(in_dir, f"subject_{int(subject_ids[0])}.csv")):
            print(f"⚠️ 데이터 없음(건너뜀): {in_dir}")
            continue

        with pd.ExcelWriter(out_path, engine='openpyxl') as writer:
            #out_path = os.path.join(out_dir, file_name)

            # # 데이터 존재 여부 확인 (에러 방지용)
            sample_file = os.path.join(in_dir, f"subject_{int(subject_ids[0])}.csv")
            if not os.path.exists(sample_file): continue
            
            # # 분석 타입(분류/회귀)에 따른 추출 지표 설정
            available_cols = pd.read_csv(sample_file).columns.tolist()
            if model_type == 'classification':
                f1_col = next((c for c in available_cols if 'f1' in c), None)
                cols = ['test_accuracy', f1_col] if f1_col else ['test_accuracy']
            else:
                cols = ['test_neg_mean_squared_error', 'test_neg_mean_absolute_error']

            # # 지표별 루프 실행 (정확도, F1, RMSE, MAE 등)
            for col in cols:
                # # 시트 이름 한글 매핑
                if 'accuracy' in col: sheet_name = "정확도"
                elif 'f1' in col: sheet_name = "F1_스코어"
                elif 'squared_error' in col: sheet_name = "RMSE_오차"
                elif 'absolute_error' in col: sheet_name = "MAE_오차"
                else: sheet_name = col

                # # 피험자별 파일 순회하며 데이터 수집
                for s_id in subject_ids:
                    in_file = os.path.join(in_dir, f"subject_{int(s_id)}.csv")
                    if os.path.exists(in_file):
                        val = pd.read_csv(in_file)[col]
                        # # 회귀 지표 변환 (음수 MSE -> 양수 RMSE, 음수 MAE -> 양수 MAE)
                        if "squared_error" in col: out_df[f"피험자_{s_id}"] = np.sqrt(-val)
                        elif "absolute_error" in col: out_df[f"피험자_{s_id}"] = -val
                        else: out_df[f"피험자_{s_id}"] = val
            
                out_df.index.name = '교차검증_폴드'
                
                # # [통계 처리] 결과 데이터 복사본에 평균 및 표준편차 행 추가
                temp_df = out_df.copy()
                temp_df.loc['평균'] = temp_df.mean()
                temp_df.loc['표준편차'] = temp_df.std()
                
                # # 엑셀 시트로 기록
                temp_df.to_excel(writer, sheet_name=sheet_name, index=True)

# ---------------------------------------------------------
# 2. 일반 모델 통합 결과 표 생성 (Generic Table)
# ---------------------------------------------------------
def generate_generic_result_table(dataset, signal, model_type):
    subject_ids = utils.get_subjects_ids(dataset=dataset)
    subjects_kor = [f"피험자_{x}" for x in subject_ids]
    in_dir = os.path.join(paths.result_directory(), "model-performance", model_type,
                          dataset, signal, "generic-model")
    
    models = utils.get_model(model_type=model_type)
    for clf in models:
        model_name = type(clf).__name__
        sample_path = os.path.join(in_dir, model_name, f"subject_{int(subject_ids[0])}.csv")
        if not os.path.exists(sample_path): continue
        
        # # 지표명 및 값 컬럼 자동 식별
        sample_data = pd.read_csv(sample_path)
        metric_col = sample_data.columns[0]
        val_col = sample_data.columns[1]
        
        # # 피험자별 성능 통합 데이터프레임 초기화
        out_df = pd.DataFrame(index=sample_data[metric_col].values, columns=subjects_kor)
        
        for s_id in subject_ids:
            in_file = os.path.join(in_dir, model_name, f"subject_{int(s_id)}.csv")
            if os.path.exists(in_file):
                data = pd.read_csv(in_file)
                out_df[f"피험자_{s_id}"] = data[val_col].values

        # # 인덱스 명칭 한글화
        out_df.index = [x.replace("Accuracy", "정확도").replace("RMSE", "RMSE_오차").replace("MAE", "MAE_오차") for x in out_df.index]
        out_df.index.name = '평가_지표'
        
        # # CSV 파일 저장 (UTF-8-SIG 인코딩으로 한글 깨짐 방지)
        out_dir = paths.ensure_directory_exists(os.path.join(paths.result_directory(), "tables",
                                                             model_type, dataset, signal, "generic", model_name))
        file_name = f"{dataset}_{signal}_{model_type}_일반모델_결과.csv"
        out_df.to_csv(os.path.join(out_dir, file_name), index=True, encoding='utf-8-sig')

# ---------------------------------------------------------
# 3. 하이브리드 보정 곡선 데이터 생성 (Calibration Table)
# ---------------------------------------------------------
def generate_calibration_table(dataset, signal, model_type):
    # # 데이터셋별 보정 샘플 크기 정보 확보
    calibration_samples = [int(x) for x in utils.get_calibration_sample_sizes(dataset=dataset) / 4]
    in_dir = os.path.join(paths.result_directory(), "model-performance", model_type, dataset, signal, "calibration")
    
    models = utils.get_model(model_type=model_type)
    for clf in models:
        model_name = type(clf).__name__
        # # 주요 분석 모델(ExtraTrees 등) 선별 적용
        if "ExtraTrees" not in model_name: continue 
        
        sample_path = os.path.join(in_dir, model_name, f"{calibration_samples[0]}.csv")
        if not os.path.exists(sample_path): continue
        
        # # 샘플 수 변화에 따른 지표 데이터 수집
        sample_data = pd.read_csv(sample_path)
        metrics = sample_data.iloc[:, 0].values
        out_df = pd.DataFrame(index=metrics, columns=calibration_samples)
        
        for sample in calibration_samples:
            in_file = os.path.join(in_dir, model_name, f"{sample}.csv")
            if os.path.exists(in_file):
                data = pd.read_csv(in_file)
                out_df[sample] = data.iloc[:, 1].values
        
        # # 최종 지표 한글 명칭 적용
        new_index = []
        for idx in out_df.index:
            if "RMSE" in idx or "squared_error" in idx: new_index.append("RMSE_오차")
            elif "MAE" in idx or "absolute_error" in idx: new_index.append("MAE_오차")
            elif "Accuracy" in idx: new_index.append("정확도")
            elif "Precision" in idx: new_index.append("정밀도")
            else: new_index.append(idx)
        out_df.index = new_index
        out_df.index.name = '평가_지표'
        
        # # 결과 저장
        out_dir = paths.ensure_directory_exists(os.path.join(paths.result_directory(), "tables",
                                                             model_type, dataset, signal, "calibration", model_name))
        file_name = f"{dataset}_{signal}_{model_type}_보정곡선_데이터.csv"
        out_df.to_csv(os.path.join(out_dir, file_name), index=True, encoding='utf-8-sig')
        
# main.py 설정과 동일해야함: 이 파일만 실행할거면 주석해제하여 실행하기
#if __name__ == "__main__":
#    datasets = ["swell"]
#   signals = ["hrv"]
#   model_types = ["regression"]
    
#   for d in datasets:
#       for s in signals:
#           for m in model_types:
#               # [개인 맞춤형 표] 피험자별 상세 결과 테이블 생성
#               generate_person_specific_result_table(d, s, m)
#               # [일반 모델 표] 전체 평균 성능 통합 테이블 생성
#               generate_generic_result_table(d, s, m)
#               # [보정 모델 표] 샘플 수 증가에 따른 성능 향상 수치 테이블 생성
#               generate_calibration_table(d, s, m)
#   print("표 데이터 업데이트 완료!")