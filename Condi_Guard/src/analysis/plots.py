# ---------------------------------------------------------
# 생체 신호 스트레스 예측 결과 시각화 모듈
# ---------------------------------------------------------
import matplotlib
# # GUI 미지원 환경(서버 등) 대응을 위한 백엔드 설정
matplotlib.use('Agg') 
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import paths
import utils

# ---------------------------------------------------------
# 1. 시각화 환경 설정 함수
# ---------------------------------------------------------
def set_korean_font():
    # # OS별 한글 폰트 설정 (Windows: 맑은 고딕)
    if os.name == 'nt':
        plt.rc('font', family='Malgun Gothic')
    # # 차트 내 마이너스 기호 깨짐 방지
    plt.rc('axes', unicode_minus=False)
    print("한글 폰트 설정(맑은 고딕)이 완료되었습니다.")

# # 초기 실행 시 폰트 설정 적용
set_korean_font()

# # 그래프 기본 규격 설정
marker_size = 4
width = 10.0 
height = 6.0 

def set_plot_style_():
    # # 시각화 테마 설정 (seaborn 우선, 실패 시 ggplot 적용)
    try:
        plt.style.use('seaborn-v0_8-paper') 
    except:
        plt.style.use('ggplot') 
    
    # # 세부 디자인 파라미터 업데이트
    # # - 선 굵기, 라벨 크기, 그리드 투명도 등 통합 관리
    settings = {
        'lines.linewidth': 1.5,
        'axes.labelsize': 10,
        'xtick.labelsize': 8,
        'ytick.labelsize': 8,
        'legend.fontsize': 9,
        'axes.titlesize': 10,
        'axes.grid': True,
        'grid.alpha': 0.3
    }
    matplotlib.rcParams.update(settings)

# ---------------------------------------------------------
# 2. 결과 출력 및 저장 함수
# ---------------------------------------------------------
def save_figure(file_name, formats=None, plot_folder=None, fig=None):
    # # 레이아웃 최적화 (요소 간 겹침 방지)
    if fig is not None:
        fig.tight_layout()
    # # 기본 저장 포맷 설정 (PNG, PDF)
    if formats is None:
        formats = ['png', 'pdf']
    # # 기본 저장 경로 설정
    if plot_folder is None:
        plot_folder = paths.plots_directory()
        
    # # 포맷별 폴더 생성 및 파일 저장
    for fmt in formats:
        out_dir = os.path.join(plot_folder, fmt)
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        plot_file = os.path.join(out_dir, f"{file_name}.{fmt}")
        # # 해상도(300DPI) 및 배경 투명도 설정 적용
        fig.savefig(plot_file, format=fmt, bbox_inches='tight', dpi=300, transparent=False)
        print(f"이미지 저장 완료: {plot_file}")

# ---------------------------------------------------------
# 3. 모델 성능 비교 그래프 (Generic vs Person-Specific)
# ---------------------------------------------------------
def plot_generic_vs_person_specific_model(dataset, signal, model_name, model_types):
    # # 대상 피험자 ID 리스트 확보
    subjects = utils.get_subjects_ids(dataset=dataset)
    set_plot_style_()
    
    # # [레이아웃 결정 로직]
    # # - model_types 개수에 따라 행(row) 수 동적 생성
    # # - 1개 타입(1단), 2개 타입(2단) 대응
    num_plots = len(model_types)
    fig, axes = plt.subplots(nrows=num_plots, ncols=1, sharex=True, 
                             figsize=(width, height if num_plots > 1 else height/2))
    
    # # - 단일 그래프일 경우 배열 인덱싱 에러 방지를 위한 리스트화
    if num_plots == 1:
        axes = [axes]

    current_idx = 0

    # # [분류 모델(Classification) 시각화 구간]
    if "classification" in model_types:
        ax = axes[current_idx]
        clf_model_name = model_name + "Classifier"
        load_file = f"{dataset}_{signal}_classification_개인별_결과.xlsx"
        gen_file = f"{dataset}_{signal}_classification_일반모델_결과.csv"
        
        try:
            # # - 데이터 경로 구성 (tables 폴더 참조)
            pers_path = os.path.normpath(os.path.join(paths.result_directory(), "tables", "classification", dataset, signal, "person-specific", clf_model_name, load_file))
            gen_path = os.path.join(paths.result_directory(), "tables", "classification", dataset, signal, "generic", clf_model_name, gen_file)
            
            # # - 개인별(Excel) 및 일반(CSV) 결과 로드
            pers_df = pd.read_excel(pers_path, sheet_name="정확도", index_col=0, engine='openpyxl')
            gen_df = pd.read_csv(gen_path, index_col=0, encoding='utf-8-sig')
            
            # # - 시각화 데이터 매핑 (X: 피험자, Y: 정확도%)
            ax.plot(subjects, (pers_df.loc["평균"] * 100).tolist(), marker='o', markersize=marker_size, color="#ca0020", label='개인 맞춤형 모델')
            ax.plot(subjects, (gen_df.loc["정확도"] * 100).tolist(), marker='o', markersize=marker_size, color="#6a3d9a", label='일반 모델')
            
            # # - 그래프 속성 설정 (Y축 범위 0~110)
            ax.set_ylim([0, 110]); ax.set_ylabel('정확도 (%)')
            ax.legend(loc='upper right')
            sns.despine(ax=ax)
        except Exception as e:
            print(f"⚠️ 분류 그래프 생성 실패: {e}")
        current_idx += 1

    # # [회귀 모델(Regression) 시각화 구간]
    if "regression" in model_types:
        ax = axes[current_idx]
        reg_model_name = model_name + "Regressor"
        load_file = f"{dataset}_{signal}_regression_개인별_결과.xlsx"
        gen_file = f"{dataset}_{signal}_regression_일반모델_결과.csv"
        
        try:
            # # - 데이터 경로 구성
            pers_path = os.path.join(paths.result_directory(), "tables", "regression", dataset, signal, "person-specific", reg_model_name, load_file)
            gen_path = os.path.join(paths.result_directory(), "tables", "regression", dataset, signal, "generic", reg_model_name, gen_file)
            
            # # - 시트 존재 여부 확인 후 적절한 시트(RMSE_오차) 로드
            xls = pd.ExcelFile(pers_path, engine='openpyxl')
            target_sheet = "RMSE_오차" if "RMSE_오차" in xls.sheet_names else xls.sheet_names[0]
            pers_df = pd.read_excel(pers_path, sheet_name=target_sheet, index_col=0, engine='openpyxl')
            gen_df = pd.read_csv(gen_path, index_col=0, encoding='utf-8-sig')
            
            # # - RMSE 지표 추출 및 매핑
            pers_rmse = pers_df.loc["평균"].tolist()
            gen_rmse = gen_df.loc[gen_df.index.str.contains("RMSE"), :].values.flatten().tolist()
            
            # # - 시각화 수행 (개인화 vs 일반 대조)
            ax.plot(subjects, pers_rmse, marker='o', markersize=marker_size, color="#ca0020", label='개인 맞춤형 모델')
            ax.plot(subjects, gen_rmse, marker='o', markersize=marker_size, color="#6a3d9a", label='일반 모델')
            ax.set_ylabel('RMSE 오차'); ax.set_xlabel('피험자 번호')
            ax.legend(loc='upper right')
            sns.despine(ax=ax)
        except Exception as e:
            print(f"⚠️ 회귀 그래프 생성 실패: {e}")

    # # 최종 이미지 파일 저장
    save_figure(file_name=f"{dataset}_{signal}_모델성능비교", fig=fig)
    plt.close()

# ---------------------------------------------------------
# 4. 하이브리드 보정 곡선 (Calibration Curve) 생성
# ---------------------------------------------------------
def plot_calibration_result(dataset, signal, model_name, model_types):
    set_plot_style_()
    
    # # [레이아웃 결정 로직]
    # # - model_types 설정값 기반으로 서브플롯 행 결정
    num_plots = len(model_types)
    fig, axes = plt.subplots(nrows=num_plots, ncols=1, sharex=True, 
                             figsize=(width, height if num_plots > 1 else height/2))
    
    # # - 단일 타입일 경우 예외 처리
    if num_plots == 1:
        axes = [axes]

    current_idx = 0

    # # [분류 보정(Classification Calibration) 시각화]
    if "classification" in model_types:
        ax = axes[current_idx]
        try:
            # # - 보정 데이터 로드 (샘플 수 증가에 따른 변화 수치)
            in_dir = os.path.join(paths.result_directory(), "tables", "classification", dataset, signal, "calibration")
            file_path = os.path.join(in_dir, model_name + "Classifier", f"{dataset}_{signal}_classification_보정곡선_데이터.csv")
            data = pd.read_csv(file_path, index_col="평가_지표", encoding='utf-8-sig')
            
            # # - X축(샘플 수), Y축(정확도/정밀도) 데이터 추출
            samples = [int(x) for x in data.columns]
            ax.plot(samples, (data.loc["정밀도"] * 100).tolist(), marker='o', color="#e41a1c", label='정밀도 (%)')
            ax.plot(samples, (data.loc["정확도"] * 100).tolist(), marker='o', color="#377eb8", label='정확도 (%)')
            
            ax.set_ylabel('분류 정확도 (%)'); ax.legend(loc='lower right')
        except Exception as e:
            print(f"⚠️ 분류 보정 그래프 생성 실패: {e}")
        current_idx += 1

    # # [회귀 보정(Regression Calibration) 시각화]
    if "regression" in model_types:
        ax = axes[current_idx]
        try:
            # # - 보정 데이터 로드
            in_dir = os.path.join(paths.result_directory(), "tables", "regression", dataset, signal, "calibration")
            file_path = os.path.join(in_dir, model_name + "Regressor", f"{dataset}_{signal}_regression_보정곡선_데이터.csv")
            data = pd.read_csv(file_path, index_col="평가_지표", encoding='utf-8-sig')
            
            # # - X축(샘플 수), Y축(오차 지표) 데이터 추출
            samples = [int(x) for x in data.columns]
            ax.plot(samples, data.loc["MAE_오차"].tolist(), marker='o', color="#4daf4a", label='MAE_ 오차')
            ax.plot(samples, data.loc["RMSE_오차"].tolist(), marker='o', color="#984ea3", label='RMSE_ 오차')
            
            # # - 데이터셋별 최적 Y축 범위 설정
            ax.set_ylim([-1, 15]) if dataset != "wesad" else ax.set_ylim([-0.1, 1.5])
            ax.set_ylabel('회귀 오차'); ax.set_xlabel('피험자당 보정 샘플 수'); ax.legend(loc='upper right')
        except Exception as e:
            print(f"⚠️ 회귀 보정 그래프 생성 실패: {e}")

    # # 최종 이미지 파일 저장
    save_figure(file_name=f"{dataset}_{signal}_보정결과", fig=fig)
    plt.close()