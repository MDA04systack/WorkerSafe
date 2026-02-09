import os
import numpy as np
import wfdb                         # PhysioNet 데이터 포맷(.dat, .hea) 처리를 위한 라이브러리
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from scipy.signal import resample

# ---------------------------------------------------------
# 0. 스타일 및 폰트 설정
# ---------------------------------------------------------
sns.set_style("whitegrid")
plt.rc('font', family='Malgun Gothic')      # 한글 폰트 설정
plt.rcParams['axes.unicode_minus'] = False

# ==================================================================================
# 1. 설정 및 경로 수정
# ==================================================================================
# workersafe.py가 호출할 때를 대비하여 상대 경로를 동적으로 설정합니다.
current_dir = os.path.dirname(os.path.abspath(__file__))
# Early_Guard.py가 찾을 수 있도록 SCA_Risk_Prediction 폴더를 저장 위치로 지정합니다.
data_path = r'D:\Semi2\INT\SCA_Risk_Prediction\dataset\sudden-cardiac-death-holter-database-1.0.0' 
save_dir = os.path.dirname(os.path.abspath(__file__))

# [중요] 환자별 심실세동(VF) 발생 실제 시각 (데이터베이스 주석 기반 수동 매핑)
# 이 시각이 곧 'Target Event' 시점이며, 예측 모델의 기준점
vf_times = {
    '30': '07:54:33', '31': '13:42:24', '32': '16:45:18', '33': '04:46:19',
    '34': '06:35:44', '35': '24:34:56', '36': '18:59:01', '37': '01:31:13',
    '38': '08:01:54', '39': '04:37:51', '41': '02:59:24', '43': '15:37:11',
    '44': '19:38:45', '46': '03:41:47', '47': '06:13:01', '48': '02:29:40',
    '50': '11:45:43', '51': '22:58:23', '52': '02:32:40'
}

target_fs = 128      # 모든 데이터의 주파수를 128Hz로 통일 (Resampling)
n_bins = 1024        # 주파수 분석(DFT) 후 사용할 피처의 개수

# ---------------------------------------------------------
# 헬퍼 함수
# ---------------------------------------------------------
def parse_vf_time_str(time_str):
    # 문자열 시간('24:34:56' 등)을 datetime 객체로 변환
    # PhysioNet 데이터에는 자정을 넘어가면 24시, 25시로 표기된 경우가 있어 이를 처리
    h, m, s = map(int, time_str.split(':'))
    days_add = 0
    if h >= 24:     # 24시 이상인 경우 날짜를 하루 더하고 시간을 보정
        days_add = h // 24
        h = h % 24
    return datetime.strptime(f"{h:02d}:{m:02d}:{s:02d}", '%H:%M:%S'), days_add

def get_real_path(base_dir, r_id):
    # 폴더 구조가 불규칙한 경우(폴더 안에 폴더가 있거나 바로 파일이 있거나)를 처리
    p1 = os.path.join(base_dir, r_id, r_id)
    if os.path.exists(p1 + ".hea"): return p1
    p2 = os.path.join(base_dir, r_id)
    if os.path.exists(p2 + ".hea"): return p2
    return None

# ==================================================================================
# 2. 시각화 함수
# ==================================================================================
def visualize_sca(record_id, segment_1min, target_fs):
    # 급성 심정지군 데이터의 마지막 구간(심실세동 발생 직전)을 시각화
    print(f"---Record {record_id}: 시각화 생성 중... (데이터 길이: {len(segment_1min)})---")
    
    # (b) 마지막 10초 데이터 추출
    # 심실세동 발생 직전의 패턴 확인용
    ten_sec_samples = 10 * target_fs
    segment_10sec = segment_1min[-ten_sec_samples:] # 심실세동 발생 전 10초
    
    # (c) DFT(주파수 변환) 계산
    # 모델의 실제 입력값
    fft_val = np.abs(np.fft.rfft(segment_1min))
    features = fft_val[:n_bins]
    
    # 그래프 생성
    fig = plt.figure(figsize=(12, 10))
    
    # (a) 심실세동 발생 1분 전 ECG 샘플
    plt.subplot(3, 1, 1)
    plt.plot(segment_1min, color='firebrick', linewidth=0.8)
    plt.title(f"(a) 급성 심정지 1분 전 ECG 샘플 ({record_id})", fontsize=12)
    plt.ylabel("신호 진폭 (mV)")
    plt.grid(True, alpha=0.3)
    
    # (b) 심실세동 발생 10초 전 ECG 샘플
    plt.subplot(3, 1, 2)
    plt.plot(segment_10sec, color='firebrick', linewidth=1.0)
    plt.title(f"(b) 급성 심정지 10초 전 ECG 샘플 ({record_id})", fontsize=12)
    plt.ylabel("신호 진폭 (mV)")
    plt.grid(True, alpha=0.3)
    
    # (c) 심실세동 발생 1분 전 주파수 스펙트럼
    plt.subplot(3, 1, 3)
    plt.plot(features, color='firebrick')
    plt.title(f"(c) 급성 심정지 1분 전 DFT 주파수 분석 ({record_id})", fontsize=12)
    plt.xlabel("주파수 빈 (frequency bins)")
    plt.ylabel("신호의 ω 주파수 성분의 크기 |F(w)|")
    plt.xlim(0, n_bins)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    # [수정] 그림을 띄우지 않고 파일로 저장
    save_path = os.path.join(save_dir, f"SCA_{record_id}_preview.png")
    plt.savefig(save_path, dpi=150) 
    plt.close() # 메모리 해제를 위해 반드시 닫아줌
    # plt.show() # 주석 처리

# ==================================================================================
# 3. 데이터 처리 함수 (ID, 시간 구간 저장)
# ==================================================================================
def process_sca_final(vf_dict, data_path):
    all_X = []      # 학습 데이터 (DFT 특징)
    all_groups = [] # 환자 ID (uuid)
    all_times = []  # 시간 정보 (60-59, ..., 1-0)
    
    print(">>> 금성 심정지 데이터 전처리 시작 (심실세동 발생 1시간 전부터 추출) <<<")
    
    #for record_id, vf_time_str in vf_dict.items():
    for record_id, vf_time_str in list(vf_dict.items())[:1]: # 드라이테스트용
        # 파일 경로 찾기
        full_path_base = get_real_path(data_path, record_id)
        if full_path_base is None:
            print(f"---{record_id}: 파일 없음---")
            continue
        
        try:
            # WFDB 로드
            record = wfdb.rdrecord(full_path_base)
            data = record.p_signal[:, 0]
            fs = record.fs
            
            # --- [핵심] 심실세동 발생 시점의 인덱스 찾기 ---
            vf_time_parsed, days_add = parse_vf_time_str(vf_time_str)
            
            # 기록 시작 시간(base_time) 설정
            base_time_dt = datetime(1900, 1, 1, 0, 0, 0)    # 기본값
            if record.base_time:
                # 실제 기록 시작 시간이 헤더에 있다면 사용
                base_time_dt = datetime.combine(vf_time_parsed.date(), record.base_time)
            
            # 심실세동 발생 시간 계산
            vf_full_dt = datetime.combine(vf_time_parsed.date(), vf_time_parsed.time()) + timedelta(days=days_add)
            
            # 시작 시간과 심실세동 발생 시간의 차이(초) 계산 -> 인덱스로 변환
            delta_seconds = (vf_full_dt - base_time_dt).total_seconds()
            if delta_seconds < 0: delta_seconds += 24 * 3600    # 날짜가 넘어간 경우 보정
            
            vf_idx = int(delta_seconds * fs)    # 심실세동 발생 시점의 샘플 인덱스
            
            # 범위 검사
            if vf_idx > len(data):
                print(f"{---record_id}: 데이터 범위 초과 (Skip)---")
                continue
            
            # 1시간 전 추출
            # 심실세동 발생 시점(vf_idx)으로부터 1시간(3600초) 전 인덱스 계산
            start_idx = max(0, vf_idx - int(3600 * fs))
            segment_1h = data[start_idx:vf_idx]
            
            # 데이터가 1분 미만이면 제외
            if len(segment_1h) < fs * 60:
                print(f"---{record_id}: 데이터 너무 짧음 (Skip)---")
                continue

            # 리샘플링: 250Hz -> 128Hz
            num_samples = int(len(segment_1h) * target_fs / fs)
            if num_samples == 0: continue
            data_resampled = resample(segment_1h, num_samples)

            # 1분 단위 분할 (Segmentation)
            samples_per_min = target_fs * 60
            num_segments = len(data_resampled) // samples_per_min
            num_segments = min(num_segments, 60)
            
            if num_segments == 0: continue

            # 세그먼트 처리
            for i in range(num_segments):
                seg = data_resampled[i*samples_per_min : (i+1)*samples_per_min]
                
                # DFT 변환
                fft_val = np.abs(np.fft.rfft(seg))
                feat = fft_val[:n_bins] if len(fft_val) >= n_bins else np.pad(fft_val, (0, n_bins-len(fft_val)), 'constant')
                all_X.append(feat)
                
                # [메타 데이터 1] uuid
                all_groups.append(record_id)
                
                # [메타 데이터 2] 시간 구간
                # i가 0일 때 (시작점): "60분 전 - 59분 전"
                # i가 59일 때 (종료점): "1분 전 - 0분 전(심실세동 발생)"
                minutes_before_start = num_segments - i
                minutes_before_end = num_segments - i - 1
                time_label = f"{minutes_before_start}-{minutes_before_end}"
                all_times.append(time_label)
                
                # [시각화] 가장 위험한 순간(마지막 1분, 심실세동 직전) 시각화
                if i == num_segments - 1:
                    visualize_sca(record_id, seg, target_fs)
            
            print(f"---Record {record_id}: 완료 ({num_segments}분)---")
            
        except Exception as e:
            print(f"---Record {record_id} 에러: {e}---")
            continue

    return np.array(all_X), np.array(all_groups), np.array(all_times)

# ==================================================================================
# 4. 실행 및 파일 저장 (파일명 수정: sca -> SCA)
# ==================================================================================
X_sca, groups_sca, times_sca = process_sca_final(vf_times, data_path)

# 샘플 수 및 환자 수 집계
extracted_count = len(X_sca)
unique_people_count = len(np.unique(groups_sca)) if extracted_count > 0 else 0

print("\n" + "=" * 50)
print(f"---급성 심정지 데이터 추출 결과 집계---")
print(f" - 총 샘플(Segment) 수 : {extracted_count}개")
print(f" - 총 환자(Record) 수  : {unique_people_count}명")
print("=" * 50)

if extracted_count > 0:
    # 라벨 생성: 급성 심정지군은 모두 1로 라벨링 (Binary Classification용)
    y_sca = np.ones(extracted_count)
    
    # Early_Guard.py의 로딩 코드와 일치하도록 대문자 'SCA'로 저장 (중요)
    np.save(os.path.join(save_dir, 'X_SCA_test.npy'), X_sca)
    np.save(os.path.join(save_dir, 'y_SCA_test.npy'), y_sca)
    np.save(os.path.join(save_dir, 'groups_SCA_test.npy'), groups_sca)
    np.save(os.path.join(save_dir, 'times_SCA_test.npy'), times_sca)
    
    print(f"---SCA 데이터 저장 완료!---")
    print(f" - 저장 경로: {save_dir}")
    print(f" - 데이터(X_SCA) 형태: {X_sca.shape}")
    print("-" * 50)
else:
    print(f"---저장할 데이터가 없습니다.---")