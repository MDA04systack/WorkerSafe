import numpy as np
import pandas as pd
import os
import wfdb
from scipy.signal import find_peaks, welch
from scipy.stats import skew, kurtosis, yeojohnson
from scipy.interpolate import interp1d
from collections import deque

# 1. R-peak 검출 함수
def detect_r_peaks_raw(signal, fs):
    diff = np.diff(signal)
    squared = diff ** 2
    window_size = int(0.12 * fs)
    mva = np.convolve(squared, np.ones(window_size)/window_size, mode='same')
    mva_peaks, _ = find_peaks(mva, distance=int(0.2 * fs))
    final_peaks = []
    search_window = int(0.05 * fs)
    for peak in mva_peaks:
        start = max(0, peak - search_window)
        end = min(len(signal), peak + search_window)
        real_peak = start + np.argmax(signal[start:end])
        final_peaks.append(real_peak)
    return np.array(final_peaks)

# 2. 안전한 변환 함수 (LOG, SQRT, YEO-JOHNSON)
def safe_log(val): return np.log(val + 1e-6) if val > 0 else 0
def safe_sqrt(val): return np.sqrt(val) if val >= 0 else 0
def safe_yj(val):
    try: return yeojohnson([val, val + 0.0001])[0][0]
    except: return val

# 3. 이미지 내 94개 변수 구성 함수
def calculate_hrv_features(rri, fs):
    if len(rri) < 10: return None
    
    rri = np.array(rri)
    rri_diff = np.diff(rri)
    rel_rri = rri_diff / rri[:-1]
    diff_rel_rri = np.diff(rel_rri)
    
    # [기본 연산 - Base 34개]
    f = {}
    f['MEAN_RR'] = np.mean(rri)
    f['MEDIAN_RR'] = np.median(rri)
    f['SDRR'] = np.std(rri, ddof=1)
    f['RMSSD'] = np.sqrt(np.mean(rri_diff**2))
    f['SDSD'] = np.std(rri_diff, ddof=1)
    f['SDRR_RMSSD'] = f['SDRR'] / f['RMSSD'] if f['RMSSD'] > 0 else 0
    f['HR'] = 60000 / f['MEAN_RR']
    f['pNN25'] = (np.sum(np.abs(rri_diff) > 25) / len(rri_diff)) * 100
    f['pNN50'] = (np.sum(np.abs(rri_diff) > 50) / len(rri_diff)) * 100
    f['SD1'] = np.sqrt(0.5 * (f['SDSD']**2))
    f['SD2'] = np.sqrt(2 * (f['SDRR']**2) - 0.5 * (f['SDSD']**2))
    f['KURT'] = kurtosis(rri)
    f['SKEW'] = skew(rri)
    f['MEAN_REL_RR'] = np.mean(rel_rri)
    f['MEDIAN_REL_RR'] = np.median(rel_rri)
    f['SDRR_REL_RR'] = np.std(rel_rri, ddof=1)
    f['RMSSD_REL_RR'] = np.sqrt(np.mean(diff_rel_rri**2))
    f['SDSD_REL_RR'] = np.std(diff_rel_rri, ddof=1)
    f['SDRR_RMSSD_REL_RR'] = f['SDRR_REL_RR'] / f['RMSSD_REL_RR'] if f['RMSSD_REL_RR'] > 0 else 0
    f['KURT_REL_RR'] = kurtosis(rel_rri)
    f['SKEW_REL_RR'] = skew(rel_rri)

    try:
        x = np.cumsum(rri) / 1000
        f_int = interp1d(x, rri, kind='linear', fill_value="extrapolate")
        x_new = np.arange(x[0], x[-1], 1/4)
        rri_res = f_int(x_new)
        freqs, psd = welch(rri_res, fs=4, nperseg=min(len(rri_res), 256))
        vlf = np.trapz(psd[(freqs >= 0.0033) & (freqs < 0.04)], freqs[(freqs >= 0.0033) & (freqs < 0.04)])
        lf = np.trapz(psd[(freqs >= 0.04) & (freqs < 0.15)], freqs[(freqs >= 0.04) & (freqs < 0.15)])
        hf = np.trapz(psd[(freqs >= 0.15) & (freqs < 0.4)], freqs[(freqs >= 0.15) & (freqs < 0.4)])
        tp = vlf + lf + hf
        f.update({'VLF': vlf, 'VLF_PCT': (vlf/tp)*100, 'LF': lf, 'LF_PCT': (lf/tp)*100, 'LF_NU': (lf/(tp-vlf))*100,
                  'HF': hf, 'HF_PCT': (hf/tp)*100, 'HF_NU': (hf/(tp-vlf))*100, 'TP': tp, 'LF_HF': lf/hf, 'HF_LF': hf/lf})
    except:
        for k in ['VLF','VLF_PCT','LF','LF_PCT','LF_NU','HF','HF_PCT','HF_NU','TP','LF_HF','HF_LF']: f[k] = 0
    
    f['sampen'], f['higuci'] = 0, 0 

    # [변환 적용 및 94개 정규화 리스트]
    final = f.copy()
    ls_targets = ['MEAN_RR', 'MEDIAN_RR', 'SDRR', 'RMSSD', 'SDSD', 'SDRR_RMSSD', 'HR', 'SD1', 'SD2', 
                  'SDRR_REL_RR', 'RMSSD_REL_RR', 'SDSD_REL_RR', 'SDRR_RMSSD_REL_RR', 'VLF', 'VLF_PCT', 
                  'LF', 'LF_PCT', 'LF_NU', 'HF', 'HF_PCT', 'HF_NU', 'TP', 'LF_HF', 'HF_LF', 'sampen', 'higuci']
    yj_targets = ['pNN25', 'pNN50', 'KURT', 'SKEW', 'MEAN_REL_RR', 'MEDIAN_REL_RR', 'KURT_REL_RR', 'SKEW_REL_RR']

    for t in ls_targets:
        final[f"{t}_LOG"] = safe_log(f[t])
        final[f"{t}_SQRT"] = safe_sqrt(f[t])
    for t in yj_targets:
        final[f"{t}_YEO_JON"] = safe_yj(f[t])

    # 이미지와 1:1 대응되는 94개 컬럼 순서
    ordered_columns = [
        'MEAN_RR', 'MEDIAN_RR', 'SDRR', 'RMSSD', 'SDSD', 'SDRR_RMSSD', 'HR', 'pNN25', 'pNN50', 'SD1', 'SD2',
        'KURT', 'SKEW', 'MEAN_REL_RR', 'MEDIAN_REL_RR', 'SDRR_REL_RR', 'RMSSD_REL_RR', 'SDSD_REL_RR',
        'SDRR_RMSSD_REL_RR', 'KURT_REL_RR', 'SKEW_REL_RR', 'VLF', 'VLF_PCT', 'LF', 'LF_PCT', 'LF_NU',
        'HF', 'HF_PCT', 'HF_NU', 'TP', 'LF_HF', 'HF_LF', 'sampen', 'higuci',
        'MEAN_RR_LOG', 'MEAN_RR_SQRT', 'MEDIAN_RR_LOG', 'MEDIAN_RR_SQRT', 'SDRR_LOG', 'SDRR_SQRT',
        'RMSSD_LOG', 'RMSSD_SQRT', 'SDSD_LOG', 'SDSD_SQRT', 'SDRR_RMSSD_LOG', 'SDRR_RMSSD_SQRT',
        'HR_LOG', 'HR_SQRT', 'pNN25_YEO_JON', 'pNN50_YEO_JON', 'SD1_LOG', 'SD1_SQRT', 'SD2_LOG', 'SD2_SQRT',
        'KURT_YEO_JON', 'SKEW_YEO_JON', 'MEAN_REL_RR_YEO_JON', 'MEDIAN_REL_RR_YEO_JON', 
        'SDRR_REL_RR_LOG', 'SDRR_REL_RR_SQRT', 'RMSSD_REL_RR_LOG', 'RMSSD_REL_RR_SQRT',
        'SDSD_REL_RR_LOG', 'SDSD_REL_RR_SQRT', 'SDRR_RMSSD_REL_RR_LOG', 'SDRR_RMSSD_REL_RR_SQRT',
        'KURT_REL_RR_YEO_JON', 'SKEW_REL_RR_YEO_JON', 'VLF_LOG', 'VLF_SQRT', 'VLF_PCT_LOG', 'VLF_PCT_SQRT',
        'LF_LOG', 'LF_SQRT', 'LF_PCT_LOG', 'LF_PCT_SQRT', 'LF_NU_LOG', 'LF_NU_SQRT', 'HF_LOG', 'HF_SQRT',
        'HF_PCT_LOG', 'HF_PCT_SQRT', 'HF_NU_LOG', 'HF_NU_SQRT', 'TP_LOG', 'TP_SQRT', 'LF_HF_LOG', 'LF_HF_SQRT',
        'HF_LF_LOG', 'HF_LF_SQRT', 'sampen_LOG', 'sampen_SQRT', 'higuci_LOG', 'higuci_SQRT'
    ]
    
    # 딕셔너리 형태로 정렬하여 반환
    return {col: final.get(col, 0) for col in ordered_columns}

# ==============================================================================
# 4. 메인 처리 로직 (경로 수정 및 자동화)
# ==============================================================================
def main():
    # [1] 경로 설정: workersafe.py 실행 위치를 기준으로 상대 경로 설정
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 데이터가 저장될 위치: Personalized_Stress_Estimator/data/swell
    output_dir = os.path.join(current_dir, "..", "data", "swell")
    output_path = os.path.join(output_dir, "hrv_features_final_test.csv")
    
    # [2] 입력 데이터 경로 (사용자 환경에 맞게 절대경로 유지 또는 수정)
    # 기존 코드에서 사용하시던 30번 레코드의 상위 폴더 경로입니다.
    input_dir = r"D:\Semi2\INT\Personalized_Stress_Estimator\dataset\sudden-cardiac-death-holter-database-1.0.0\30"
    
    # [3] 신호 읽기 (path 정의 포함)
    path = os.path.join(input_dir, "30") # 파일명 '30'을 포함한 전체 경로
    
    try:
        print(f"Data loading from: {path}")
        record = wfdb.rdrecord(path)
        signal = record.p_signal[:, 0]
        fs = record.fs
        
        # RR-Interval 산출 및 피처 추출 로직
        peaks = detect_r_peaks_raw(signal, fs)
        rri_total = np.diff(peaks) / fs * 1000 
        
        window_ms = 5 * 60 * 1000 
        current_window = deque()
        current_window_sum = 0
        results = []

        print("🧪 피처 추출 시작 (5분 윈도우)...")
        for i, new_ibi in enumerate(rri_total):
            current_window.append(new_ibi)
            current_window_sum += new_ibi
            while current_window_sum > window_ms:
                current_window_sum -= current_window.popleft()
                
            if current_window_sum >= (window_ms * 0.9):
                if i % 100 == 0: # 100샘플마다 추출
                    feat = calculate_hrv_features(list(current_window), fs)
                    if feat:
                        results.append(feat)

        # DataFrame 생성 및 CSV 저장
        df = pd.DataFrame(results)
        os.makedirs(output_dir, exist_ok=True)
        
        # Condi_Guard 가 읽기 쉽도록 index 없이 저장
        df.to_csv(output_path, index=False, encoding='utf-8-sig')
        
        print(f"Success: ECG Preprocessing completed at {output_path}")
        
    except Exception as e:
        print(f"Error: ECG Preprocessing failed - {e}")

if __name__ == "__main__":
    main()