import numpy as np
import pandas as pd
from scipy.signal import find_peaks, welch, butter, filtfilt
from scipy.stats import skew, kurtosis, yeojohnson
from scipy.interpolate import interp1d
import os

# ==============================================================================
# [단계 1] 전처리 및 피크 검출 함수
# ==============================================================================

def ppg_bandpass_filter(signal, fs):
    """0.5Hz ~ 8.0Hz 대역 통과 필터 (고주파 및 저주파 노이즈 동시 제거)"""
    nyq = 0.5 * fs
    low, high = 0.5 / nyq, 8.0 / nyq
    b, a = butter(4, [low, high], btype='band')
    return filtfilt(b, a, signal)

def detect_ppg_peaks(signal, fs):
    """필터링된 신호에서 심박 피크 검출"""
    min_dist = int(0.4 * fs)  # 최소 0.4초 간격 (성인 심박수 고려)
    peaks, _ = find_peaks(signal, distance=min_dist, prominence=np.std(signal) * 0.5)
    return peaks

# ==============================================================================
# [단계 2] 비선형 지표 계산 함수 (Higuchi, SampEn)
# ==============================================================================

def calculate_higuchi(X, k_max=10):
    """히구치 프랙탈 차원 (Complexity 측정)"""
    N = len(X)
    if N < k_max: return 0
    L_k = []
    for k in range(1, k_max + 1):
        Lk_m = []
        for m in range(k):
            Lmk = 0
            n_intervals = int((N - m - 1) / k)
            if n_intervals == 0: continue
            for i in range(1, n_intervals + 1):
                Lmk += abs(X[m + i * k] - X[m + (i - 1) * k])
            norm = (N - 1) / (n_intervals * k)
            Lk_m.append((Lmk * norm) / k)
        if Lk_m: L_k.append(np.mean(Lk_m))
    
    if len(L_k) < 2: return 0
    ln_k = np.log(1. / np.arange(1, len(L_k) + 1))
    ln_Lk = np.log(L_k)
    return np.polyfit(ln_k, ln_Lk, 1)[0]

def calculate_sampen(L, m=2, r=0.2):
    """샘플 엔트로피 (불규칙성 측정)"""
    L = np.array(L)
    N = len(L)
    if N <= m: return 0
    r_val = r * np.std(L)
    if r_val == 0: return 0
    def _phi(m_len):
        x = np.array([L[i:i+m_len] for i in range(N-m_len+1)])
        C = 0
        for i in range(len(x)):
            dist = np.max(np.abs(x - x[i]), axis=1)
            C += np.sum(dist <= r_val) - 1
        return C
    phi_m, phi_m1 = _phi(m), _phi(m+1)
    return -np.log(phi_m1 / phi_m) if phi_m != 0 and phi_m1 != 0 else 0

# ==============================================================================
# [단계 3] 94개 피처 추출 메인 함수
# ==============================================================================

def calculate_hrv_94_features(rri, fs):
    """RRI 기반 시간, 주파수, 비선형 변환 포함 94개 피처 산출"""
    if len(rri) < 10: return None
    
    rri = np.array(rri)
    rri_diff = np.diff(rri)
    rel_rri = 2 * (rri_diff / (rri[1:] + rri[:-1])) 
    diff_rel_rri = np.diff(rel_rri)
    
    f = {}
    # 기본 지표 계산
    f['MEAN_RR'], f['MEDIAN_RR'] = np.mean(rri), np.median(rri)
    f['SDRR'] = np.std(rri, ddof=1)
    f['RMSSD'] = np.sqrt(np.mean(rri_diff**2))
    f['SDSD'] = np.std(rri_diff, ddof=1)
    f['SDRR_RMSSD'] = f['SDRR'] / f['RMSSD'] if f['RMSSD'] > 0 else 0
    f['HR'] = 60000 / f['MEAN_RR']
    f['pNN25'] = (np.sum(np.abs(rri_diff) > 25) / len(rri_diff)) * 100
    f['pNN50'] = (np.sum(np.abs(rri_diff) > 50) / len(rri_diff)) * 100
    f['SD1'] = np.sqrt(0.5 * (f['SDSD']**2))
    f['SD2'] = np.sqrt(2 * (f['SDRR']**2) - 0.5 * (f['SDSD']**2))
    f['KURT'], f['SKEW'] = kurtosis(rri), skew(rri)
    f['MEAN_REL_RR'], f['MEDIAN_REL_RR'] = np.mean(rel_rri), np.median(rel_rri)
    f['SDRR_REL_RR'] = np.std(rel_rri, ddof=1)
    f['RMSSD_REL_RR'] = np.sqrt(np.mean(diff_rel_rri**2))
    f['SDSD_REL_RR'] = np.std(diff_rel_rri, ddof=1)
    f['SDRR_RMSSD_REL_RR'] = f['SDRR_REL_RR'] / f['RMSSD_REL_RR'] if f['RMSSD_REL_RR'] > 0 else 0
    f['KURT_REL_RR'], f['SKEW_REL_RR'] = kurtosis(rel_rri), skew(rel_rri)

    # 주파수 영역 분석
    try:
        x_t = np.cumsum(rri) / 1000
        f_int = interp1d(x_t, rri, kind='linear', fill_value="extrapolate")
        x_new = np.arange(x_t[0], x_t[-1], 1/4) 
        rri_res = f_int(x_new)
        freqs, psd = welch(rri_res, fs=4, nperseg=min(len(rri_res), 256))
        vlf = np.trapz(psd[(freqs >= 0.0033) & (freqs < 0.04)], freqs[(freqs >= 0.0033) & (freqs < 0.04)])
        lf = np.trapz(psd[(freqs >= 0.04) & (freqs < 0.15)], freqs[(freqs >= 0.04) & (freqs < 0.15)])
        hf = np.trapz(psd[(freqs >= 0.15) & (freqs < 0.4)], freqs[(freqs >= 0.15) & (freqs < 0.4)])
        tp = vlf + lf + hf
        f.update({
            'VLF': vlf, 'VLF_PCT': (vlf/tp)*100 if tp>0 else 0, 'LF': lf, 'LF_PCT': (lf/tp)*100 if tp>0 else 0, 
            'LF_NU': (lf/(tp-vlf))*100 if (tp-vlf)>0 else 0, 'HF': hf, 'HF_PCT': (hf/tp)*100 if tp>0 else 0, 
            'HF_NU': (hf/(tp-vlf))*100 if (tp-vlf)>0 else 0, 'TP': tp, 'LF_HF': lf/hf if hf>0 else 0, 'HF_LF': hf/lf if lf>0 else 0
        })
    except:
        for k in ['VLF','VLF_PCT','LF','LF_PCT','LF_NU','HF','HF_PCT','HF_NU','TP','LF_HF','HF_LF']: f[k] = 0
    
    f['sampen'], f['higuci'] = calculate_sampen(rri), calculate_higuchi(rri)

    # 데이터 변환 (LOG, SQRT, YEO-JOHNSON)
    final = f.copy()
    ls_targets = ['MEAN_RR', 'MEDIAN_RR', 'SDRR', 'RMSSD', 'SDSD', 'SDRR_RMSSD', 'HR', 'SD1', 'SD2', 
                  'SDRR_REL_RR', 'RMSSD_REL_RR', 'SDSD_REL_RR', 'SDRR_RMSSD_REL_RR', 'VLF', 'VLF_PCT', 
                  'LF', 'LF_PCT', 'LF_NU', 'HF', 'HF_PCT', 'HF_NU', 'TP', 'LF_HF', 'HF_LF', 'sampen', 'higuci']
    yj_targets = ['pNN25', 'pNN50', 'KURT', 'SKEW', 'MEAN_REL_RR', 'MEDIAN_REL_RR', 'KURT_REL_RR', 'SKEW_REL_RR']

    for t in ls_targets:
        val = f.get(t, 0)
        final[f"{t}_LOG"] = np.log(val + 1e-6) if val > 0 else 0
        final[f"{t}_SQRT"] = np.sqrt(val) if val >= 0 else 0
    for t in yj_targets:
        val = f.get(t, 0)
        try: final[f"{t}_YEO_JON"] = yeojohnson([val + 1, val + 1.1])[0][0]
        except: final[f"{t}_YEO_JON"] = val

    return final

# ==============================================================================
# [단계 4] 분석 실행 루틴 (Time Feature 제거 버전)
# ==============================================================================

def process_ppg_analysis(raw_signal, fs, window_size_min=5, step_size_min=1):
    filtered = ppg_bandpass_filter(raw_signal, fs)
    peaks = detect_ppg_peaks(filtered, fs)
    
    peak_times_sec = peaks / fs
    win_size_sec = window_size_min * 60
    step_sec = step_size_min * 60
    
    all_results = []
    current_start = 0
    max_time = peak_times_sec[-1] if len(peak_times_sec) > 0 else 0

    while current_start + win_size_sec <= max_time or (current_start == 0 and max_time > 0):
        mask = (peak_times_sec >= current_start) & (peak_times_sec < current_start + win_size_sec)
        win_peaks = peaks[mask]
        
        if len(win_peaks) >= 10:
            ppi = np.diff(win_peaks) / fs * 1000
            ppi = ppi[(ppi > 400) & (ppi < 1500)] 
            
            if len(ppi) >= 10:
                feats = calculate_hrv_94_features(ppi, fs)
                if feats:
                    # START_TIME_SEC, END_TIME_SEC 추가 로직 제거
                    all_results.append(feats)
        
        if current_start + win_size_sec > max_time: break
        current_start += step_sec
    
    return pd.DataFrame(all_results)

# ==============================================================================
# [실행부] (경로 자동화 및 Condi_Guard 연동)
# ==============================================================================

if __name__ == "__main__":
    # [1] 경로 설정: 현재 파일 위치 기준
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Stress 모델(Condi_Guard)이 데이터를 찾는 표준 경로로 설정
    # Personalized_Stress_Estimator/data/swell
    save_dir = os.path.join(current_dir, "..", "data", "swell")
    os.makedirs(save_dir, exist_ok=True)

    # [2] 입력 데이터 경로 (사용자 환경 설정)
    file_directory = r'D:\Semi2\INT\Personalized_Stress_Estimator\dataset\heartbeatfatigue'
    file_name = 'gamer1-ppg-5min.csv'
    full_path = os.path.join(file_directory, file_name)

    try:
        # 파일 존재 확인
        if not os.path.exists(full_path):
            raise FileNotFoundError(f"입력 파일을 찾을 수 없습니다: {full_path}")

        df = pd.read_csv(full_path)
        signal_array = df['Red_Signal'].values
        SAMPLING_RATE = 75.08 
        
        print(f"📡 {file_name} PPG 분석 시작...")
        
        # 5분 단위 윈도우로 특징 추출
        final_df = process_ppg_analysis(signal_array, SAMPLING_RATE, window_size_min=5)
        
        if not final_df.empty:
            # Condi_Guard와 연동하기 위한 파일명 설정
            output_name = os.path.join(save_dir, 'ppg_features_final_test.csv')
            final_df.to_csv(output_name, index=False, encoding='utf-8-sig')
            
            print(f" PPG 전처리 완료!")
            print(f" - 저장 위치: {output_name}")
            print(f" - 추출된 피처 형태: {final_df.shape}")
        else:
            print("분석 실패: 유효한 심박 구간을 확보하지 못했습니다.")

    except Exception as e:
        print(f" PPG 전처리 중 오류 발생: {e}")