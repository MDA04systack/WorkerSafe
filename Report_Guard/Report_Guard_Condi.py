import os
import sys
import warnings
import joblib
import pandas as pd
import numpy as np

# TensorFlow 로그 억제 및 경고 무시
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings('ignore')

# 1. 경로 설정 (원본 파일의 경로 로직 유지)
current_dir = os.path.dirname(os.path.abspath(__file__))
int_root = os.path.abspath(os.path.join(current_dir, '..'))

# 시스템 경로 등록
src_path = os.path.join(int_root, "Early_Guard", "src")
xai_path = os.path.join(src_path, "xai")
for p in [int_root, src_path, xai_path, current_dir]:
    if p not in sys.path:
        sys.path.insert(0, p)

import paths
try:
    import config
except ImportError:
    # 만약 다른 경로에서 호출될 경우를 대비한 상대 경로 임포트
    from . import config
    
from stress_analyzer import run_stress_xai # 업로드하신 stress_analyzer.py 활용

class StressAnalyzer:
    """스트레스 데이터 로드 및 XAI 분석을 수행하는 클래스"""
    
    def __init__(self):
        self.int_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.model_path = paths.get_model_path()
        self.model = joblib.load(self.model_path)
        
        # 1. 경로 정의 (경로를 먼저 선언해야 에러가 안 납니다)
        all_samples_path = os.path.join(paths.data_directory(), "hrv", "swell", "all-samples.csv")
        calib_path = os.path.join(paths.data_directory(), "hrv", "swell", "calibration.csv")

        # 2. 데이터 로드
        bank_df = pd.read_csv(all_samples_path) 
        calib_df = pd.read_csv(calib_path) 

        # 3. [핵심] 데이터셋 역할 분리
        # (1) 비교 은행: DisCERN이 "정상 상태"를 찾을 수 있게 전체 데이터셋 활용
        # '위험'한 상태인 나에게 '정상'인 타인의 지표를 제안하기 위함입니다.
        self.train_df = pd.concat([bank_df, calib_df], axis=0).reset_index(drop=True)
        
        # (2) 전시용 명단: 사이드바에 띄울 보정 데이터셋 (AttributeError 해결)
        self.display_df = calib_df.copy() 
        self.display_df['subject_id'] = self.display_df['subject_id'].astype(str) # ID 문자열화
        
        # 4. 피처 동기화
        if hasattr(self.model, 'feature_names_in_'):
            self.actual_features = list(self.model.feature_names_in_)
        else:
            self.actual_features = config.HRV_FEATURES[:75]

        print(f"✅ 배경 데이터 로드: {len(self.train_df)}건 (비교 은행)")
        print(f"✅ 전시용 명단 로드: {len(self.display_df)}건 (보정 데이터)")
    
    def get_analysis_results(self, input_row):
        sample_input = input_row[self.actual_features].values
        res = run_stress_xai(self.model, sample_input, self.train_df, self.actual_features)
        
        # [핵심] 게이지 점수를 데이터의 nasa_tlx 값으로 설정
        orig_score = input_row['NasaTLX']
        
        # config.py의 Grier(2015) 기준 적용 (33, 57, 60)
        if orig_score >= config.CONDI_THRESHOLDS["HIGH_LIMIT"]:
            status_key = "HIGH"
        elif orig_score >= config.CONDI_THRESHOLDS["LOW_LIMIT"]:
            status_key = "MEDIUM"
        else:
            status_key = "LOW"
        
        guide_data = config.CONDI_GUIDE[status_key]
        
        # 3. 결과 데이터 구성 (기존 로직 유지)
        df = pd.DataFrame({
            '항목': self.actual_features,
            '현재_norm': res[0], '추천_norm': res[1],
            '현재수치': res[4], '목표수치': res[5],
            'SHAP_importance': res[6]
        })
        
        # 영어 이름을 한글 이름으로 치환
        df['항목'] = df['항목'].map(lambda x: config.HRV_NAME_KOR.get(x, x))
        
        top_reason = df.sort_values(by='SHAP_importance', ascending=False).iloc[0]['항목']
        
        return {
            'original_score': orig_score,
            'target_score': res[3], # CF가 제안하는 목표 점수(라벨)
            'status': guide_data['title'],
            'status_key': status_key,
            'interpretation': f"{guide_data['desc']} 특히 '{top_reason}' 지표 관리가 필요합니다.",
            'guide': guide_data['action'],
            'full_df': df,
            'changes': df[(df['추천_norm'] - df['현재_norm']).abs() > 1e-4], 
            'top_reason': top_reason,
            'color': guide_data['color'] # 시각화용 색상 추가
        }