### 급성 심정지(SCA) 위험 예측 1D-CNN 모델 실행 코드

import os # 현재 실행 위치인 INT 폴더를 기준으로 상대 경로 설정
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # 화면 출력 없이 파일 저장만 가능하게 설정
import seaborn as sns
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc, accuracy_score, recall_score, precision_score, f1_score

# ==================================================================================
# 0. 스타일 및 폰트 설정
# ==================================================================================
# 시각화 그래프의 배경 스타일과 한글 폰트 깨짐 방지를 위한 설정
sns.set_style("whitegrid")
plt.rc('font', family='Malgun Gothic')
plt.rcParams['axes.unicode_minus'] = False  # 마이너스 기호(-) 깨짐 방지

# ==================================================================================
# 1. 데이터 로드 및 uuid 기준 분리
#    메타 데이터 보존
# ==================================================================================

# # 현재 작업 디렉토리(CWD)를 기준으로 경로 설정
# load_dir = r'D:/Semi2/INT/SCA_Risk_Prediction'
base_path = os.getcwd()
load_dir = os.path.join(base_path, 'SCA_Risk_Prediction')
save_dir = os.path.join(load_dir, 'saved_models')

print("--- 데이터 로딩 및 개인 ID 기준 분리 중...")

# 1. 파일 로드
# X: 신호데이터, y: 라벨, g: uuid, t: 시간정보
try:
    # 정상군(Normal) 데이터 로드
    X_normal = np.load(os.path.join(load_dir, 'X_normal.npy'))
    y_normal = np.load(os.path.join(load_dir, 'y_normal.npy'))
    g_normal = np.load(os.path.join(load_dir, 'groups_normal.npy')) # uuid
    t_normal = np.load(os.path.join(load_dir, 'times_normal.npy'))  # 시간 구간
    
    # 급성 심정지군 (SCA - 급성 심정지) 데이터 로드
    X_SCA = np.load(os.path.join(load_dir, 'X_SCA.npy'))
    y_SCA = np.load(os.path.join(load_dir, 'y_SCA.npy'))
    g_SCA = np.load(os.path.join(load_dir, 'groups_SCA.npy'))       # uuid
    t_SCA = np.load(os.path.join(load_dir, 'times_SCA.npy'))        # 시간 구간
except Exception as e:
    print(f"--- 데이터 로드 실패: {e}")
    exit()

# ==================================================================================
# 2. 개인 ID 기준 분리 함수 (메타 데이터 g, t 포함)
# ==================================================================================
# 그룹 기반 분할
# [중요] 의료 데이터에서는 한 개인의 데이터가 Train과 Test에 섞여 들어가면 
# 모델이 개인의 고유 특성을 외워버리는 '데이터 누수(Data Leakage)'가 발생
# 이를 방지하기 위해 반드시 개인 uuid를 기준으로 데이터를 분할
# 80% 학습, 20% 테스트
def split_group_wise(X, y, groups, times, group_name="Data", test_ratio=0.2):
    unique_ids = np.unique(groups) # 중복 제거된 고유 개인 ID 목록 추출
    
    # 개인 ID 리스트를 무작위로 Train/Test 그룹으로 분할
    train_ids, test_ids = train_test_split(unique_ids, test_size=test_ratio, random_state=42)
    
    # 분할 결과 출력
    print(f"\n--- [{group_name}] 그룹 분할 결과")
    print(f" - 전체 인원: {len(unique_ids)}명")
    print(f" - Train ({len(train_ids)}명): {sorted(train_ids)}")
    print(f" - Test  ({len(test_ids)}명): {sorted(test_ids)}")
    
    # 전체 데이터에서 Train ID에 속하는지, Test ID에 속하는지 마스킹(True/False) 배열 생성
    train_mask = np.isin(groups, train_ids)
    test_mask = np.isin(groups, test_ids)
    
    # 마스킹을 사용하여 실제 데이터 분할 및 반환
    # (Test셋은 시각화를 위해 개인 ID와 시간정보도 함께 반환)
    return (X[train_mask], X[test_mask], 
            y[train_mask], y[test_mask], 
            groups[test_mask], times[test_mask]) # 테스트셋의 메타정보 반환

# ==================================================================================
# 3. 분리 수행 (정상군과 위험군 각각 분리)
# ==================================================================================
X_train_n, X_test_n, y_train_n, y_test_n, g_test_n, t_test_n = split_group_wise(
    X_normal, y_normal, g_normal, t_normal, group_name="정상군(Normal)")

X_train_s, X_test_s, y_train_s, y_test_s, g_test_s, t_test_s = split_group_wise(
    X_SCA, y_SCA, g_SCA, t_SCA, group_name="급성 심정지군(SCA)")

# ==================================================================================
# 4. 데이터 합치기 및 개별 파일 저장 (.npy & .csv)
# ==================================================================================

# [추가] 데이터셋 저장용 폴더 설정
dataset_dir = os.path.join(load_dir, 'dataset')
if not os.path.exists(dataset_dir):
    os.makedirs(dataset_dir)

# 1) 데이터 합치기
X_train = np.concatenate((X_train_n, X_train_s), axis=0)
y_train = np.concatenate((y_train_n, y_train_s), axis=0)
X_test = np.concatenate((X_test_n, X_test_s), axis=0)
y_test = np.concatenate((y_test_n, y_test_s), axis=0)

g_test_combined = np.concatenate((g_test_n, g_test_s), axis=0)  # 개인 ID
t_test_combined = np.concatenate((t_test_n, t_test_s), axis=0)  # 시간 정보

# 2) NPY 파일 저장 (XAI 알고리즘 및 모델 입력용)
# XAI 알고리즘이 개별 파일을 읽도록 설정되어 있다면 이 방식이 가장 안전합니다.
np.save(os.path.join(dataset_dir, 'X_train_final.npy'), X_train)
np.save(os.path.join(dataset_dir, 'y_train_final.npy'), y_train)
np.save(os.path.join(dataset_dir, 'X_test_final.npy'), X_test)
np.save(os.path.join(dataset_dir, 'y_test_final.npy'), y_test)
np.save(os.path.join(dataset_dir, 'g_test_final.npy'), g_test_combined)
np.save(os.path.join(dataset_dir, 't_test_final.npy'), t_test_combined)

# 3) CSV 파일 저장 (사용자 확인용)
# 학습셋 라벨 확인용 CSV
pd.DataFrame({'label': y_train}).to_csv(os.path.join(dataset_dir, 'train_labels_check.csv'), index=False)

# 테스트셋 메타데이터 확인용 (ID + 시간 + 실제라벨)
test_check_df = pd.DataFrame({
    'uuid': g_test_combined,
    'time_info': t_test_combined,
    'target_label': y_test
})
test_check_df.to_csv(os.path.join(dataset_dir, 'test_metadata_check.csv'), index=False, encoding='utf-8-sig')

print(f"--- [NPY/CSV] 모든 데이터셋이 '{dataset_dir}' 폴더에 저장되었습니다.")
# ==================================================================================
# 5. 차원 확장 (CNN 입력용)
# ==================================================================================
# CNN(Conv1D)은 입력으로 (샘플수, 시퀀스길이, 채널수)의 3차원 형태를 요구
# 현재 (N, 1024) 형태를 (N, 1024, 1) 형태로 변환
X_train = np.expand_dims(X_train, axis=-1)
X_test_cnn = np.expand_dims(X_test, axis=-1) # 변수명 X_test_cnn으로 맞춤

print(f"--- 데이터 준비 완료")
print(f" - Train Shape: {X_train.shape}")
print(f" - Test Shape : {X_test_cnn.shape}")

# ==================================================================================
# 6. 시각화용 DataFrame 생성
# ==================================================================================
# 예측이 끝난 후, 어떤 환자의 어떤 시간대 데이터가 위험하게 예측되었는지 분석하기 위함
test_full = pd.DataFrame({
    'uuid': g_test_combined,
    'Target': y_test,   # 0: 정상, 1: 급성 심정지
    'Time_before_VF': t_test_combined
})

# ==================================================================================
# 7. 1D-CNN 모델 구축 및 학습, 객체 저장
# ==================================================================================
def build_final_model(input_shape):
    model = models.Sequential(name="SCA_Prediction_Model")
    model.add(layers.Input(shape=input_shape))
    
    # [특징 추출] Conv1D: 시계열 데이터의 지역적 패턴(파형 특징)을 학습
    model.add(layers.Conv1D(filters=64, kernel_size=8, strides=8, padding='same'))
    model.add(layers.BatchNormalization())      # 학습 안정화 및 가속
    model.add(layers.ReLU())                    # 비선형성 추가
    model.add(layers.MaxPooling1D(pool_size=2)) # 주요 특징만 남기고 차원 축소
    
    # [분류기] 추출된 특징을 바탕으로 최종 확률 계산
    model.add(layers.Flatten())                         # 1차원 벡터로 변환
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dropout(0.3))                      # 과적합(Overfitting) 방지
    model.add(layers.Dense(1, activation='sigmoid'))    # 이진 분류 (0~1 사이 확률 출력)
                                                        # 0.5 이상 = 급성 심정지 위험
    
    # 학습 설정: 학습률
    opt = tf.keras.optimizers.Adam(learning_rate=0.0013)
    
    model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
    return model

model = build_final_model(input_shape=(1024, 1))

print("/n--- 모델 학습 시작...")
# 조기 종료 설정: 검증 손실(val_loss)이 10번의 에포크 동안 개선되지 않으면 학습 중단 및 최적 모델 복구
early_stopping = callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

history = model.fit(
    X_train, y_train,
    epochs=50,              # 최대 50번 반복
    batch_size=32,          # 한 번에 32개씩 학습
    validation_data=(X_test_cnn, y_test),
    callbacks=[early_stopping],
    verbose=1
)

# 모델 저장 폴더 생성 및 저장
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

model_path = os.path.join(save_dir, 'SCA_1D_CNN_Model.keras')
model.save(model_path)

print(f"/n--- 모델 객체 파일이 저장되었습니다: {model_path}")


# ==================================================================================
# 9. 결과 시각화 및 평가 (Performance Evaluation)
# ==================================================================================
print("/n--- 상세 시각화 생성 중...")

# 예측값 생성 (확률값 & 0/1 클래스값)
y_pred_prob = model.predict(X_test_cnn).ravel() # 확률 (0.0 ~ 1.0)
y_pred = (y_pred_prob > 0.5).astype(int)        # 클래스 (0 또는 1)


# [추가] 결과 저장용 폴더 생성
result_dir = os.path.join(load_dir, 'results') # 데이터 폴더 내 results 폴더 생성
if not os.path.exists(result_dir):
    os.makedirs(result_dir)

# ==================================================================================
# 9. 결과 시각화 및 평가 (Performance Evaluation)
# ==================================================================================
print("/--- 상세 시각화 생성 및 저장 중...")

# 예측값 생성
y_pred_prob = model.predict(X_test_cnn).ravel()
y_pred = (y_pred_prob > 0.5).astype(int)

# ----------------------------------------------------------------------------------
# --- Figure 1 & 2: 학습 곡선 ---
# ----------------------------------------------------------------------------------
fig, ax = plt.subplots(1, 2, figsize=(14, 5))

ax[0].plot(history.history['loss'], label='학습 (Train)', linewidth=2, color='royalblue')
ax[0].plot(history.history['val_loss'], label='검증 (Test)', linewidth=2, linestyle='--', color='darkorange')
ax[0].set_title('Figure 1. 손실 곡선 (Loss Curve)', fontsize=14, fontweight='bold')
ax[0].set_xlabel('에포크 (Epochs)')
ax[0].set_ylabel('손실값 (Loss)')
ax[0].legend(fontsize=11)
ax[0].grid(True, alpha=0.3)

ax[1].plot(history.history['accuracy'], label='학습 (Train)', linewidth=2, color='royalblue')
ax[1].plot(history.history['val_accuracy'], label='검증 (Test)', linewidth=2, linestyle='--', color='green')
ax[1].set_title('Figure 2. 정확도 곡선 (Accuracy Curve)', fontsize=14, fontweight='bold')
ax[1].set_xlabel('에포크 (Epochs)')
ax[1].set_ylabel('정확도 (Accuracy)')
ax[1].legend(fontsize=11)
ax[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(result_dir, '01_Learning_Curves.png'), dpi=300, bbox_inches='tight')
# plt.show() # 주석 처리
plt.close()

# ----------------------------------------------------------------------------------
# --- Figure 3: 혼동 행렬(Confusion matrix) ---
# ----------------------------------------------------------------------------------
plt.figure(figsize=(6, 5))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, annot_kws={"size": 16},
            xticklabels=['정상 (Normal)', '급성 심정지 (SCA)'], yticklabels=['정상 (Normal)', '급성 심정지 (SCA)'])
plt.title('Figure 3. 혼동 행렬 (Confusion Matrix)', fontsize=14, fontweight='bold', pad=15)
plt.xlabel('예측값 (Predicted)', fontsize=12)
plt.ylabel('실제값 (Actual)', fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(result_dir, '02_Confusion_Matrix.png'), dpi=300, bbox_inches='tight')
# plt.show() # 주석 처리
plt.close()

# ----------------------------------------------------------------------------------
# --- Figure 4: ROC 커브 ---
# ----------------------------------------------------------------------------------
fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(7, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2.5, label=f'AUC = {roc_auc:.4f}')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('FPR (1 - 특이도)', fontsize=12)
plt.ylabel('민감도 (TPR)', fontsize=12)
plt.title('Figure 4. ROC 곡선', fontsize=14, fontweight='bold', pad=15)
plt.legend(loc="lower right", fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(result_dir, '03_ROC_Curve.png'), dpi=300, bbox_inches='tight')
# plt.show() # 주석 처리
plt.close()

# ----------------------------------------------------------------------------------
# --- Figure 5: 정상군 vs 위험군 시간대별 위험도 분석 (Boxplot) ---
# ----------------------------------------------------------------------------------
# (데이터 정렬 및 전처리 코드는 기존과 동일)
SCA_full_order = [f"{i}-{i-1}" for i in range(60, 0, -1)]
norm_full_order = [f"{i}-{i+1}" for i in range(0, 60)]
test_plot_df = test_full.copy()
test_plot_df['Prob_SCA'] = y_pred_prob
existing_SCA = set(test_plot_df[test_plot_df['Target'] == 1]['Time_before_VF'].unique())
existing_norm = set(test_plot_df[test_plot_df['Target'] == 0]['Time_before_VF'].unique())
SCA_order = [label for label in SCA_full_order if label in existing_SCA]
norm_order = [label for label in norm_full_order if label in existing_norm]

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20, 12), sharex=False)

# [상단] 급성 심정지군
SCA_df = test_plot_df[test_plot_df['Target'] == 1]
if not SCA_df.empty:
    sns.boxplot(x='Time_before_VF', y='Prob_SCA', data=SCA_df, order=SCA_order, color='crimson', ax=ax1)
    ax1.axhline(0.5, color='black', linestyle='--')
    ax1.set_title('급성 심정지군: 심실세동 1시간 전 → 직전', fontsize=15, fontweight='bold')
    ax1.tick_params(axis='x', rotation=45)

# [하단] 정상 대조군
norm_df = test_plot_df[test_plot_df['Target'] == 0]
if not norm_df.empty:
    sns.boxplot(x='Time_before_VF', y='Prob_SCA', data=norm_df, order=norm_order, color='royalblue', ax=ax2)
    ax2.axhline(0.5, color='black', linestyle='--')
    ax2.set_title('정상 대조군: 측정 1시간 후 기록 → 1시간 경과', fontsize=15, fontweight='bold')
    ax2.tick_params(axis='x', rotation=90)

plt.tight_layout()
plt.savefig(os.path.join(result_dir, '04_Time_Risk_Analysis.png'), dpi=300, bbox_inches='tight')
# plt.show() # 주석 처리
plt.close()

# ==================================================================================
# --- 성능 지표 출력 ---
# ==================================================================================
tn, fp, fn, tp = cm.ravel()
metrics_df = pd.DataFrame({
    '지표 (Metric)': ['정확도 (Accuracy)', '민감도 (Sensitivity)', '특이도 (Specificity)', '정밀도 (Precision)', 'F1 점수', 'AUC'],
    'Value': [
        accuracy_score(y_test, y_pred),
        recall_score(y_test, y_pred),
        tn / (tn + fp) if (tn + fp) > 0 else 0,
        precision_score(y_test, y_pred),
        f1_score(y_test, y_pred),
        roc_auc
    ]
})

print("/n[ --- 최종 모델 성능 평가  --- ]")
print(metrics_df.round(4))

# [추가] 지표 결과도 CSV로 저장
metrics_df.to_csv(os.path.join(result_dir, 'model_performance_metrics.csv'), index=False, encoding='utf-8-sig')
print(f"--- 모든 시각화 결과와 성능 지표가 {result_dir} 폴더에 저장되었습니다.")

# 해석:
# 정확도(Accuracy): 전체 데이터(급성 심정지 + 정상인) 중에서 모델이 정답을 맞힌 비율
# 민감도 (Sensitivity): 실제 심정지 환자 중에서 모델이 "위험하다"고 올바르게 찾아낸 비율
# 특이도 (Specificity): 실제 정상인 중에서 모델이 "정상이다"라고 올바르게 분류한 비율
# 정밀도 (Precision): 모델이 "위험하다(SCA)"라고 경고한 사람들 중, 실제로 환자인 비율
# F1 점수 (F1-Score): 민감도와 정밀도의 조화 평균으로, 두 지표의 균형을 보여주는 점수
# AUC (Area Under Curve): 민감도와 특이도의 관계를 종합하여, 모델의 전반적인 구별 능력(변별력)을 점수화
