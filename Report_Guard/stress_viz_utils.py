## 스트레스 전용 스트림릿 UI (SHAP 대응 버전)
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np

# --------------------------------------------------------------------
# 함수 이름: display_stress_xai_results
# 목적: SHAP 기여도와 DisCERN 기반 행동 가이드를 출력함
# --------------------------------------------------------------------
def display_stress_xai_results(feature_names, original_instance, cf_instance, original_score, target_score, shap_values=None, real_original=None, real_cf=None):
    st.markdown("---")
    st.header("🧘 AI 개인 맞춤형 스트레스 관리 가이드")

    # 1. 요약 섹션 (현재 점수 vs 목표 점수)
    col1, col2, col3 = st.columns(3)
    with col1:
        st.error(f"**현재 스트레스 점수**\n\n {original_score:.2f}")
    with col2:
        st.success(f"**목표 개선 점수**\n\n {target_score:.2f}")
    with col3:
        changed_count = sum(1 for i in range(len(feature_names)) if abs(original_instance[i] - cf_instance[i]) > 1e-4)
        st.info(f"**필요 조치 항목**\n\n {changed_count}개 생체 지표")

    # 데이터 정리
    df = pd.DataFrame({
        '항목': feature_names,
        '현재_norm': original_instance,
        '추천_norm': cf_instance,
        '현재수치': real_original if real_original is not None else original_instance,
        '목표수치': real_cf if real_cf is not None else cf_instance
    })
    
    if shap_values is not None:
        df['SHAP_importance'] = shap_values

    # 변화가 필요한 항목 추출
    changes = df[ (df['추천_norm'] - df['현재_norm']).abs() > 1e-4 ].copy()

    # 2. SHAP 원인 분석 섹션 (Why?)
    # SHAP을 쓰기로 했으므로, 지표별 기여도를 시각화합니다.
    st.subheader("1️⃣ 스트레스 유발 요인 분석 (SHAP)")
    st.write("각 지표가 현재 스트레스 점수를 높이는 데 기여한 정도입니다.")
    
    if shap_values is not None:
        # 기여도 기준 내림차순 정렬
        importance_df = df.sort_values(by='SHAP_importance', ascending=True)
        fig_shap = go.Figure(go.Bar(
            x=importance_df['SHAP_importance'],
            y=importance_df['항목'],
            orientation='h',
            marker_color='royalblue'
        ))
        fig_shap.update_layout(
            xaxis_title="스트레스 점수 기여도 (오른쪽일수록 위험 요소)",
            height=300,
            margin=dict(l=20, r=20, t=20, b=20)
        )
        st.plotly_chart(fig_shap, use_container_width=True)
    else:
        st.markdown(" ".join([f"**`{name}`**" for name in changes['항목']]))
    
    # 3. 행동 가이드 섹션 (How?)
    st.subheader("2️⃣ 상태 개선을 위한 행동 지침 (DisCERN)")
    
        
    fig = go.Figure()
    for i, row in changes.iterrows():
        fig.add_annotation(
            x=row['추천_norm'], y=row['항목'], ax=row['현재_norm'], ay=row['항목'],
            xref="x", yref="y", axref="x", ayref="y",
            showarrow=True, arrowhead=3, arrowsize=1, arrowwidth=3, arrowcolor="green"
        )
    
    fig.add_trace(go.Scatter(
        x=changes['현재_norm'], y=changes['항목'], 
        mode='markers', name='현재 상태', 
        marker=dict(color='red', size=12)
    ))
    
    fig.add_trace(go.Scatter(
        x=changes['추천_norm'], y=changes['항목'], 
        mode='markers', name='개선 목표', 
        marker=dict(color='green', size=14, symbol='star')
    ))
    
    fig.update_layout(
        xaxis_title="지표 변화 방향 (정규화 수치)", 
        height=300 + (len(changes) * 40),
        margin=dict(l=20, r=20, t=20, b=20)
    )
    st.plotly_chart(fig, use_container_width=True)

    # 4. 상세 수치 테이블
    st.write("📋 **생체 신호 조정 상세 가이드:**")
    guide_df = changes[['항목', '현재수치', '목표수치']].copy()
    
    def format_val(v):
        return f"{v:.2f}" if isinstance(v, (float, np.float32, np.float64)) else v

    guide_df['현재수치'] = guide_df['현재수치'].apply(format_val)
    guide_df['목표수치'] = guide_df['목표수치'].apply(format_val)
    
    guide_df.columns = ['측정 항목', '현재 수치', '개선 목표 수치']
    st.table(guide_df)

    st.info("💡 **전문가 제언:** 분석 결과, 상단 차트에서 기여도가 높게 나타난 지표를 우선적으로 개선하는 것이 효과적입니다.")