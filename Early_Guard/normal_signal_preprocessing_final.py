import os
import numpy as np
import wfdb                         # 의료 데이터 포맷(.dat, .hea)을 읽기 위한 라이브러리
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.signal import resample   # 주파수 변경(Resampling)을 위한 함수

# ---------------------------------------------------------
# 0. 스타일 및 폰트 설정
# ---------------------------------------------------------
sns.set_style("whitegrid")
plt.rc('font', family='Malgun Gothic')
plt.rcParams['axes.unicode_minus'] = False  # 마이너스 기호 깨짐 방지

# ==================================================================================
# 1. 설정 및 경로
# ==================================================================================
# 원본 데이터가 있는 폴더와 처리된 데이터를 저장할 폴더를 지정
current_dir = os.path.dirname(os.path.abspath(__file__))
save_dir = current_dir # SCA 전처리와 동일한 폴더로 지정
data_path = r'D:\Semi2\INT\SCA_Risk_Prediction\dataset\mit-bih-normal-sinus-rhythm-database-1.0.0'

# 정상군 ID 리스트 (MIT-BIH 데이터베이스 파일명과 일치)
uuid = [
    '16265', '16272', '16273', '16420', '16483', '16539', 
    '16773', '16786', '16795', '17052', '17453', '18177', 
    '18184', '19088', '19090', '19093', '19140', '19830'
    ]

# ==================================================================================
# 2. 시각화 함수
# ==================================================================================
def visualize_normal(record_id, segment_1min, target_fs, time_label):
    # 특정 구간의 ECG 신호를 시각화하여 데이터가 정상적으로 처리되었는지 확인
    # (a) 1분 파형, (b) 10초 파형, (c) 주파수(DFT) 분석 결과
    print(f"--- Record {record_id} ({time_label}분 구간): 시각화 생성 중...")
    
    n_bins = 1024   # 주파수 분석 시 사용할 frequency bin의 개수 (X축 범위)
    
    # (b) 10초 데이터 추출
    ten_sec_samples = 10 * target_fs
    segment_10sec = segment_1min[:ten_sec_samples] 
    
    # (c) DFT(이산 푸리에 변환) 계산
    # 시간 도메인의 신호를 주파수 도메인으로 변환하여 특징을 추출
    # np.fft.rfft: 실수 입력에 대한 FFT (대칭성을 이용해 계산 효율 증가)
    fft_val = np.abs(np.fft.rfft(segment_1min))
    features = fft_val[:n_bins]     # 모델 입력 크기에 맞춰 자름
    
    # 그래프 생성
    plt.figure(figsize=(12, 10))
    
    # [Subplot 1] 1분 전체 ECG
    plt.subplot(3, 1, 1)
    plt.plot(segment_1min, color='tab:blue', linewidth=0.8)
    plt.title(f"(a) 정상군 1분 ECG 샘플 ({record_id}) - 1시간 경과 후 {time_label}분 구간", fontsize=12)
    plt.ylabel("신호 진폭 (mV)")
    plt.grid(True, alpha=0.3)
    
    # [Subplot 2] 10초 확대 ECG (P-Q-R-S-T 파형 확인용)
    plt.subplot(3, 1, 2)
    plt.plot(segment_10sec, color='tab:blue', linewidth=1.0)
    plt.title(f"(b) 정상군 10초 ECG 샘플 ({record_id})", fontsize=12)
    plt.ylabel("신호 진폭 (mV)")
    plt.grid(True, alpha=0.3)
    
    # [Subplot 3] 주파수 분석 결과 (모델의 실제 입력 데이터)
    plt.subplot(3, 1, 3)
    plt.plot(features, color='tab:blue')
    plt.title(f"(c) 정상군 DFT 주파수 분석 ({record_id})", fontsize=12)
    plt.xlabel("주파수 빈 (frequency bins)")
    plt.ylabel("신호의 ω 주파수 성분의 크기 |F(w)|")
    plt.xlim(0, n_bins)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    # [수정] 그림을 띄우지 않고 파일로 저장
    save_path = os.path.join(save_dir, f"Normal_{record_id}_preview.png")
    plt.savefig(save_path, dpi=150)
    plt.close() # 메모리 해제를 위해 반드시 닫아줌
    # plt.show() # 주석 처리

# ==================================================================================
# 3. 데이터 처리 함수 (ID, Time 추가 저장)
# ==================================================================================
def process_normal_specific_hour(uuid_list, data_path):
    # 전체 환자 데이터를 순회하며 다음 과정을 수행
    # 1. 데이터 로드 -> 2. 특정 시간대 추출(1시간 후) -> 3. 리샘플링 -> 4. 1분 단위 분할 -> 5. DFT 변환
    all_X = []      # 주파수 데이터
    all_groups = [] # 그룹 정보 (개인 ID - 데이터 누수 방지용)
    all_times = []  # 시계열 정보 (몇 번째 분인지 - 시간대별 분석용)
    
    target_fs = 128     # 목표 주파수 (Hz)
    n_bins = 1024       # 주파수 특징 차원 수
    
    # 시간 설정
    # 초기 1시간은 안정화 등을 이유로 건너뛰고, 그 다음 1시간을 사용
    skip_seconds = 60 * 60       
    duration_seconds = 60 * 60   
    
    print(f"\n[설정] 시작 {skip_seconds//60}분 후부터, {duration_seconds//60}분간 데이터를 추출합니다.")
    
    # 헬퍼 함수 추가 (SCA 코드에서 가져옴)
    def get_real_path(base_dir, r_id):
        p1 = os.path.join(base_dir, r_id, r_id)
        if os.path.exists(p1 + ".hea"): return p1
        p2 = os.path.join(base_dir, r_id)
        if os.path.exists(p2 + ".hea"): return p2
        return None
    
    for record_id in uuid_list:
            # [수정] get_real_path를 사용하여 폴더 구조에 상관없이 .hea 파일을 찾음 [cite: 5]
            full_path = get_real_path(data_path, record_id)
            
            if full_path is None:
                # [수정] 인코딩 에러 방지를 위해 이모지 대신 텍스트 사용 
                print(f"[Error] {record_id}: 파일을 찾을 수 없습니다.")
                continue

            # wfdb 라이브러리를 사용해 헤더(.hea)와 데이터(.dat) 로드 [cite: 5]
            try:
                record = wfdb.rdrecord(full_path)
                data = record.p_signal[:, 0]    # 첫 번째 채널(Lead)만 사용 [cite: 5]
                fs = record.fs                  # 원본 샘플링 주파수 [cite: 5]
                
                # 1. 인덱스 슬라이싱 (안정화 시간 skip 후 duration만큼 추출) [cite: 5]
                start_idx = int(skip_seconds * fs)
                end_idx = int((skip_seconds + duration_seconds) * fs)
                
                # 데이터 길이가 부족할 경우 예외 처리 [cite: 5]
                if len(data) < end_idx:
                    print(f"[Warning] {record_id}: 데이터가 짧아 가능한 만큼만 사용합니다.")
                    segment_1h = data[start_idx:]
                else:
                    segment_1h = data[start_idx : end_idx]
                
                if len(segment_1h) == 0: 
                    continue

                # 2. 리샘플링 (Resampling): 목표 주파수인 128Hz로 통일 [cite: 5]
                num_samples_resampled = int(len(segment_1h) * target_fs / fs)
                data_resampled = resample(segment_1h, num_samples_resampled)
                
                # 3. 1분 단위 분할 (Segmentation) [cite: 5]
                samples_per_segment = target_fs * 60 
                num_segments = len(data_resampled) // samples_per_segment
                num_segments = min(num_segments, 60) # 최대 60분까지만 제한 [cite: 5]
                
                for i in range(num_segments):
                    # 1분 데이터 추출 [cite: 5]
                    seg = data_resampled[i*samples_per_segment : (i+1)*samples_per_segment]
                    
                    # DFT 변환: 주파수 도메인 특징 추출 [cite: 5]
                    fft_val = np.abs(np.fft.rfft(seg))
                    
                    # 데이터 길이를 n_bins(1024)로 고정 (부족하면 패딩) [cite: 5]
                    features = fft_val[:n_bins] if len(fft_val) >= n_bins else np.pad(fft_val, (0, n_bins-len(fft_val)), 'constant')
                    
                    all_X.append(features)
                    
                    # [메타 데이터 저장 1] 개인 ID (데이터 누수 방지용 그룹핑) [cite: 5]
                    all_groups.append(record_id)
                    
                    # [메타 데이터 저장 2] 시간 정보 ("0-1", "1-2" 등) [cite: 5]
                    time_label = f"{i}-{i+1}"
                    all_times.append(time_label)

                    # [시각화] 마지막 구간에 대해 시각화 수행 [cite: 5]
                    if i == num_segments - 1:
                        visualize_normal(record_id, seg, target_fs, time_label)
                
                print(f"[Success] Record {record_id}: 완료 ({num_segments}분)")
                
            except Exception as e:
                # [수정] 인코딩 에러 방지를 위해 텍스트로 에러 출력 
                print(f"[Error] Skipping {record_id} due to: {e}")
                continue
                
    return np.array(all_X), np.array(all_groups), np.array(all_times)

# ==================================================================================
# 4. 실행 및 저장
# ==================================================================================
# 함수 실행하여 데이터 추출
X_normal, groups_normal, times_normal = process_normal_specific_hour(uuid, data_path)

# 추출된 샘플 수 및 사람 수 계산
extracted_count = len(X_normal)
unique_people_count = len(np.unique(groups_normal)) if extracted_count > 0 else 0

print("\n" + "=" * 50)
print(f"--- 정상군 데이터 추출 결과 집계 ---")
print(f" - 총 샘플 수: {extracted_count}개")
print(f" - 총 사람 수: {unique_people_count}명")
print("=" * 50)

if extracted_count > 0:
    # 라벨 생성: 정상군은 모두 0으로 라벨링
    y_normal = np.zeros(extracted_count)
    
    # NumPy 배열(.npy) 형태로 저장
    np.save(os.path.join(save_dir, 'X_normal_test.npy'), X_normal)
    np.save(os.path.join(save_dir, 'y_normal_test.npy'), y_normal)
    np.save(os.path.join(save_dir, 'groups_normal_test.npy'), groups_normal)
    np.save(os.path.join(save_dir, 'times_normal_test.npy'), times_normal)
    
    print(f"--- Normal 데이터 저장 완료! ---")
    print(f" - 저장 경로: {save_dir}")
    print(f" - X_normal shape: {X_normal.shape}")
    print("-" * 50)
else:
    print(f"--- 저장할 데이터가 없습니다.--- ")