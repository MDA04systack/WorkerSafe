### config.py: XAI(DisCERN) 전용 데이터 허브

# 절대 경로 대신 파일 위치를 기준으로 경로를 자동 계산하도록 설정함(paths.py 참조)
import paths

# ---------------------------------------------------
# 1. 모델 객체 경로 (paths.py에 정의된 상수를 그대로 사용)
# ---------------------------------------------------
MODEL_PATH = paths.CALIBRATED_MODEL_PATH

# ---------------------------------------------------
# 2. 참조 데이터셋 경로
# ---------------------------------------------------

DATA_PATH = paths.CALIBRATION_DATA_PATH
REPORT_OUT_DIR = paths.report_output_directory()

# ---------------------------------------------------
# 3. 피처 정의서
# ---------------------------------------------------

HRV_FEATURES = [
    'MEAN_RR', 'MEDIAN_RR', 'SDRR', 'RMSSD', 'SDSD', 'SDRR_RMSSD', 'HR', 'pNN25', 'pNN50', 
    'SD1', 'SD2', 'KURT', 'SKEW', 'MEAN_REL_RR', 'MEDIAN_REL_RR', 'SDRR_REL_RR', 
    'RMSSD_REL_RR', 'SDSD_REL_RR', 'SDRR_RMSSD_REL_RR', 'KURT_REL_RR', 'SKEW_REL_RR', 
    'VLF', 'VLF_PCT', 'LF', 'LF_PCT', 'LF_NU', 'HF', 'HF_PCT', 'HF_NU', 'TP', 'LF_HF', 
    'HF_LF', 'sampen', 'higuci', 'MEAN_RR_LOG', 'MEAN_RR_SQRT', 'MEDIAN_RR_LOG', 
    'MEDIAN_RR_SQRT', 'SDRR_LOG', 'SDRR_SQRT', 'RMSSD_LOG', 'RMSSD_SQRT', 'SDSD_LOG', 
    'SDSD_SQRT', 'SDRR_RMSSD_LOG', 'SDRR_RMSSD_SQRT', 'HR_LOG', 'HR_SQRT', 'pNN25_YEO_JON', 
    'pNN50_YEO_JON', 'SD1_LOG', 'SD1_SQRT', 'SD2_LOG', 'SD2_SQRT', 'KURT_YEO_JON', 
    'SKEW_YEO_JON', 'MEAN_REL_RR_YEO_JON', 'MEDIAN_REL_RR_YEO_JON', 'SDRR_REL_RR_LOG', 
    'SDRR_REL_RR_SQRT', 'RMSSD_REL_RR_LOG','RMSSD_REL_RR_SQRT', 'SDSD_REL_RR_LOG', 
    'SDSD_REL_RR_SQRT', 'SDRR_RMSSD_REL_RR_LOG', 'SDRR_RMSSD_REL_RR_SQRT', 
    'KURT_REL_RR_YEO_JON', 'SKEW_REL_RR_YEO_JON', 'VLF_LOG', 'VLF_SQRT', 'VLF_PCT_LOG', 
    'VLF_PCT_SQRT', 'LF_LOG', 'LF_SQRT', 'LF_PCT_LOG', 'LF_PCT_SQRT', 'LF_NU_LOG', 
    'LF_NU_SQRT', 'HF_LOG', 'HF_SQRT', 'HF_PCT_LOG', 'HF_PCT_SQRT', 'HF_NU_LOG', 
    'HF_NU_SQRT', 'TP_LOG', 'TP_SQRT', 'LF_HF_LOG', 'LF_HF_SQRT', 'HF_LF_LOG', 
    'HF_LF_SQRT', 'sampen_LOG', 'sampen_SQRT', 'higuci_LOG', 'higuci_SQRT', 
    'NasaTLX', 'subject_id', 'condition'
]
# ---------------------------------------------------
# 4. Report Guard 결과 해석 및 행동 가이드
# ---------------------------------------------------

# 4-1. Condi Guard (스트레스) 가이드: NASA-TLX 글로벌 노름(Grier, 2015) 기준 설정
# 논문 출처: Grier, R. A. (2015, September). How high is high? A meta-analysis of NASA-TLX global workload scores. In Proceedings of the human factors and ergonomics society annual meeting (Vol. 59, No. 1, pp. 1727-1731). Sage CA: Los Angeles, CA: Sage Publications.
# 논문 위치: D:\Semi2\INT\Personalized_Stress_Estimator\How High Is High_ A Meta-Analysis of NASA-TLX Global Workload Scores.pdf

CONDI_THRESHOLDS = {
    "LOW_LIMIT": 33,    # 하위 25% (쾌적)
    "HIGH_LIMIT": 57,   # 상위 25% (주의 필요)
    "DANGER_ZONE": 60   # 오류 발생률 급증 구간
}

CONDI_GUIDE = {
    "HIGH": {
        "title": "🚨 작업 부하 높음 (High Load)",
        "desc": "상위 25%에 해당하는 높은 부하 상태입니다. Grier(2015) 기준에 따라 작업 환경 개선이 권장됩니다.",
        "action": [
            "작업 우선순위를 재조정하여 인지적 과부하를 줄이세요.",
            "동료와의 협업을 통해 업무를 분담하는 것을 권장합니다.",
            "충분한 휴식 시간을 확보하여 피로 누적을 방지하세요."
        ],
        "color": "red"
    },
    "MEDIUM": {
        "title": "⚠️ 작업 부하 보통 (Medium Load)",
        "desc": "평균적인 작업 부하 수준입니다. 장시간 지속될 경우 피로가 쌓일 수 있으니 주의하세요.",
        "action": [
            "정기적인 스트레칭으로 신체적 긴장을 완화하세요.",
            "현재의 작업 속도를 유지하며 컨디션을 체크하세요."
        ],
        "color": "orange"
    },
    "LOW": {
        "title": "✅ 작업 부하 낮음 (Low Load)",
        "desc": "하위 25%에 해당하는 매우 쾌적한 상태입니다. 업무 몰입에 적합한 컨디션입니다.",
        "action": [
            "현재의 안정적인 상태를 유지하며 업무에 집중하세요.",
            "최적의 업무 효율을 낼 수 있는 시기입니다."
        ],
        "color": "green"
    }
}


# HRV 지표 한글 매핑 정의 (시각화 및 리포트용)
HRV_NAME_KOR = {
    'MEAN_RR': 'RR 간격의 평균값',
    'MEDIAN_RR': 'RR 간격의 중앙값',
    'SDRR': 'RR 간격의 표준편차',
    'RMSSD': '인접한 RR 간격 차이의 제곱평균제곱근',
    'SDSD': '인접한 RR 간격 차이의 표준편차',
    'SDRR_RMSSD': 'RR 간격 표준편차 대비 인접 차이 비율',
    'HR': '심박수',
    'pNN25': '연속 RR 간격 차이가 25ms 이상인 비율',
    'pNN50': '연속 RR 간격 차이가 50ms 이상인 비율',
    'SD1': '단기 심박 변이도',
    'SD2': '장기 심박 변이도',
    'KURT': 'RR 간격 분포의 첨도',
    'SKEW': 'RR 간격 분포의 왜도',
    'MEAN_REL_RR': '상대 RR 변화율 평균',
    'MEDIAN_REL_RR': '상대 RR 변화율 중앙값',
    'SDRR_REL_RR': '상대 RR 변화율 표준편차',
    'RMSSD_REL_RR': '인접한 상대 RR 변화율 차이의 제곱평균제곱근',
    'SDSD_REL_RR': '인접한 상대 RR 변화율 차이의 표준편차',
    'SDRR_RMSSD_REL_RR': '상대 RR 기반의 변동 대비 인접 차이 비율',
    'KURT_REL_RR': '상대 RR 분포의 첨도',
    'SKEW_REL_RR': '상대 RR 분포의 왜도',
    'VLF': '초저주파',
    'VLF_PCT': '초저주파 성분 비율',
    'LF': '저주파',
    'LF_PCT': '저주파 성분 비율',
    'LF_NU': '정규화된 저주파',
    'HF': '고주파',
    'HF_PCT': '고주파 성분 비율',
    'HF_NU': '정규화된 고주파',
    'TP': '전체 주파수 전력',
    'LF_HF': '저주파/고주파 비율',
    'HF_LF': '고주파/저주파 비율',
    'sampen': '샘플 엔트로피',
    'higuci': '히구치 프렉탈 차원',
    'MEAN_RR_LOG': 'RR 간격의 평균값 (Log)',
    'MEAN_RR_SQRT': 'RR 간격의 평균값 (SQRT)',
    'MEDIAN_RR_LOG': 'RR 간격의 중앙값 (Log)',
    'MEDIAN_RR_SQRT': 'RR 간격의 중앙값 (SQRT)',
    'SDRR_LOG': 'RR 간격의 표준편차 (Log)',
    'SDRR_SQRT': 'RR 간격의 표준편차 (SQRT)',
    'RMSSD_LOG': '인접한 RR 간격 차이의 제곱평균제곱근 (Log)',
    'RMSSD_SQRT': '인접한 RR 간격 차이의 제곱평균제곱근 (SQRT)',
    'SDSD_LOG': '인접한 RR 간격 차이의 표준편차 (Log)',
    'SDSD_SQRT': '인접한 RR 간격 차이의 표준편차 (SQRT)',
    'SDRR_RMSSD_LOG': 'RR 간격 표준편차 대비 인접 차이 비율 (Log)',
    'SDRR_RMSSD_SQRT': 'RR 간격 표준편차 대비 인접 차이 비율 (SQRT)',
    'HR_LOG': '심박수 (Log)',
    'HR_SQRT': '심박수 (SQRT)',
    'pNN25_YEO_JON': '연속 RR 간격 차이가 25ms 이상인 비율 (Yeo-Johnson)',
    'pNN50_YEO_JON': '연속 RR 간격 차이가 50ms 이상인 비율 (Yeo-Johnson)',
    'SD1_LOG': '단기 심박 변이도 (Log)',
    'SD1_SQRT': '단기 심박 변이도 (SQRT)',
    'SD2_LOG': '장기 심박 변이도 (Log)',
    'SD2_SQRT': '장기 심박 변이도 (SQRT)',
    'KURT_YEO_JON': 'RR 간격 분포의 첨도 (Yeo-Johnson)',
    'SKEW_YEO_JON': 'RR 간격 분포의 왜도 (Yeo-Johnson)',
    'MEAN_REL_RR_YEO_JON': '상대 RR 변화율 평균 (Yeo-Johnson)',
    'MEDIAN_REL_RR_YEO_JON': '상대 RR 변화율 중앙값 (Yeo-Johnson)',
    'SDRR_REL_RR_LOG': '상대 RR 변화율 표준편차 (Log)',
    'SDRR_REL_RR_SQRT': '상대 RR 변화율 표준편차 (SQRT)',
    'RMSSD_REL_RR_LOG': '인접한 상대 RR 변화율 차이의 제곱평균제곱근 (Log)',
    'RMSSD_REL_RR_SQRT': '인접한 상대 RR 변화율 차이의 제곱평균제곱근 (SQRT)',
    'SDSD_REL_RR_LOG': '인접한 상대 RR 변화율 차이의 표준편차 (Log)',
    'SDSD_REL_RR_SQRT': '인접한 상대 RR 변화율 차이의 표준편차 (SQRT)',
    'SDRR_RMSSD_REL_RR_LOG': '상대 RR 기반의 변동 대비 인접 차이 비율 (Log)',
    'SDRR_RMSSD_REL_RR_SQRT': '상대 RR 기반의 변동 대비 인접 차이 비율 (SQRT)',
    'KURT_REL_RR_YEO_JON': '상대 RR 분포의 첨도 (Yeo-Johnson)',
    'SKEW_REL_RR_YEO_JON': '상대 RR 분포의 왜도 (Yeo-Johnson)',
    'VLF_LOG': '초저주파 (Log)',
    'VLF_SQRT': '초저주파 (SQRT)',
    'VLF_PCT_LOG': '초저주파 성분 비율 (Log)',
    'VLF_PCT_SQRT': '초저주파 성분 비율 (SQRT)',
    'LF_LOG': '저주파 (Log)',
    'LF_SQRT': '저주파 (SQRT)',
    'LF_PCT_LOG': '저주파 성분 비율 (Log)',
    'LF_PCT_SQRT': '저주파 성분 비율 (SQRT)',
    'LF_NU_LOG': '정규화된 저주파 (Log)',
    'LF_NU_SQRT': '정규화된 저주파 (SQRT)',
    'HF_LOG': '고주파 (Log)',
    'HF_SQRT': '고주파 (SQRT)',
    'HF_PCT_LOG': '고주파 성분 비율 (Log)',
    'HF_PCT_SQRT': '고주파 성분 비율 (SQRT)',
    'HF_NU_LOG': '정규화된 고주파 (Log)',
    'HF_NU_SQRT': '정규화된 고주파 (SQRT)',
    'TP_LOG': '전체 주파수 전력 (Log)',
    'TP_SQRT': '전체 주파수 전력 (SQRT)',
    'LF_HF_LOG': '저주파/고주파 비율 (Log)',
    'LF_HF_SQRT': '저주파/고주파 비율 (SQRT)',
    'HF_LF_LOG': '고주파/저주파 비율 (Log)',
    'HF_LF_SQRT': '고주파/저주파 비율 (SQRT)',
    'sampen_LOG': '샘플 엔트로피 (Log)',
    'sampen_SQRT': '샘플 엔트로피 (SQRT)',
    'higuci_LOG': '히구치 프렉탈 차원 (Log)',
    'higuci_SQRT': '히구치 프렉탈 차원 (SQRT)',
    'NasaTLX': 'NASA 작업 부하 지수(TLX)',
    'subject_id': '피험자 ID',
    'condition': '스트레스 상태'
}