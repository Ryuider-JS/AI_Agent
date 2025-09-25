"""
4일차: Streamlit 대시보드 개발
=====================================

이 모듈은 1-3일차에 개발한 모델들을 통합하여 
대화형 웹 대시보드를 구현합니다.

주요 기능:
1. 데이터 탐색 및 시각화
2. 실시간 예측
3. 모델 성능 비교
4. AI Assistant 통합

Author: Dashboard Development Team
Date: 2024
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pickle
import os
import warnings
from datetime import datetime, timedelta
import json

# 커스텀 모듈 import
from agent import ManufacturingAIAgent, PredictionResult

warnings.filterwarnings('ignore')

# =============================================================================
# 1. 페이지 설정
# =============================================================================

st.set_page_config(
    page_title="제조업 예측 유지보수 대시보드",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/your-repo',
        'Report a bug': "https://github.com/your-repo/issues",
        'About': "# 제조업 예측 유지보수 시스템\n4일차 프로젝트 - AI Agent & Dashboard"
    }
)

# =============================================================================
# 2. 스타일 설정
# =============================================================================

# CSS 스타일
st.markdown("""
<style>
    /* 메인 헤더 스타일 */
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        padding: 1rem;
        background: linear-gradient(90deg, #f0f2f6 0%, #e0e5eb 100%);
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    
    /* 메트릭 카드 스타일 */
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    /* 성공/경고/위험 표시 */
    .status-normal { color: green; }
    .status-warning { color: orange; }
    .status-danger { color: red; }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# 3. 세션 상태 초기화
# =============================================================================

if 'page' not in st.session_state:
    st.session_state.page = 'Home'

if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = False

if 'ai_agent' not in st.session_state:
    st.session_state.ai_agent = None

if 'data' not in st.session_state:
    st.session_state.data = None

if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False

if 'current_question' not in st.session_state:
    st.session_state.current_question = ""

if 'models' not in st.session_state:
    st.session_state.models = {}

if 'scaler' not in st.session_state:
    st.session_state.scaler = None

# =============================================================================
# 4. 유틸리티 함수
# =============================================================================

@st.cache_data
def load_data():
    """데이터 로드 (캐시 사용)"""
    try:
        df = pd.read_csv("../data/ai4i2020.csv")
        # 컬럼명 변경 (XGBoost 호환)
        column_rename_map = {
            "Air temperature [K]": "Air_temperature_K",
            "Process temperature [K]": "Process_temperature_K",
            "Rotational speed [rpm]": "Rotational_speed_rpm",
            "Torque [Nm]": "Torque_Nm",
            "Tool wear [min]": "Tool_wear_min",
        }
        df.rename(columns=column_rename_map, inplace=True)
        return df
    except Exception as e:
        st.error(f"데이터 로드 실패: {e}")
        return None

@st.cache_resource
def load_models():
    """모델 로드 (캐시 사용)"""
    models = {}
    model_files = {
        'Random Forest': '../models/rf_model.pkl',
        'XGBoost': '../models/xgb_model.pkl',
        'Deep Learning': '../models/dl_model.h5'
    }
    
    for name, path in model_files.items():
        if os.path.exists(path):
            try:
                if path.endswith('.h5'):
                    # TensorFlow 모델 로드
                    import tensorflow as tf
                    models[name] = tf.keras.models.load_model(path)
                else:
                    # Scikit-learn/XGBoost 모델 로드
                    with open(path, 'rb') as f:
                        models[name] = pickle.load(f)
                st.success(f"✅ {name} 모델 로드 완료")
            except Exception as e:
                st.warning(f"⚠️ {name} 모델 로드 실패: {e}")
    
    # 스케일러 로드
    scaler = None
    if os.path.exists('../models/scaler.pkl'):
        with open('../models/scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
    
    return models, scaler

def get_prediction(model, model_name, features, scaler=None):
    """모델 예측 수행"""
    try:
        # 스케일링
        if scaler:
            features_scaled = scaler.transform(features)
        else:
            features_scaled = features
        
        # 예측
        if model_name == 'Deep Learning':
            prediction = (model.predict(features_scaled) > 0.5).astype(int)
            probability = model.predict(features_scaled)[0][0]
        else:
            prediction = model.predict(features_scaled)
            probability = model.predict_proba(features_scaled)[0][1]
        
        return prediction[0], probability
    except Exception as e:
        st.error(f"예측 실패: {e}")
        return None, None

# =============================================================================
# 5. 사이드바
# =============================================================================

def render_sidebar():
    """사이드바 렌더링"""
    with st.sidebar:
        st.markdown("## 🏭 제조업 예측 유지보수")
        st.markdown("---")
        
        # 페이지 선택
        page = st.selectbox(
            "📍 페이지 선택",
            ["🏠 Home", "📊 데이터 분석", "🎯 실시간 예측", 
             "📈 모델 비교", "🤖 AI Assistant", "📚 학습 자료"]
        )
        st.session_state.page = page.split()[1]  # 이모지 제거
        
        st.markdown("---")
        
        # 모델 상태
        st.markdown("### 📦 모델 상태")
        if st.session_state.model_loaded:
            st.success("✅ 모델 로드 완료")
        else:
            if st.button("모델 로드", type="primary"):
                with st.spinner("모델 로딩 중..."):
                    models, scaler = load_models()
                    st.session_state.models = models
                    st.session_state.scaler = scaler
                    st.session_state.model_loaded = True
        
        st.markdown("---")
        
        # 정보
        st.markdown("### ℹ️ 정보")
        st.info(
            """
            **버전**: 1.0.0  
            **업데이트**: 2024.01  
            **라이선스**: MIT
            """
        )

# =============================================================================
# 6. 페이지별 컨텐츠
# =============================================================================

def page_home():
    """홈 페이지"""
    st.markdown('<h1 class="main-header">🏭 제조업 예측 유지보수 대시보드</h1>', 
                unsafe_allow_html=True)
    
    # 소개
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("""
        ### 시스템 소개
        
        이 대시보드는 **AI4I 2020 Predictive Maintenance Dataset**을 활용하여
        제조 장비의 고장을 예측하고 유지보수 전략을 수립하는 시스템입니다.
        
        #### 주요 기능:
        - 📊 **데이터 분석**: EDA 및 패턴 분석
        - 🎯 **실시간 예측**: 다양한 ML/DL 모델 활용
        - 📈 **모델 비교**: 성능 메트릭 비교 분석
        - 🤖 **AI Assistant**: GPT-4 기반 인사이트 제공
        """)
    
    st.markdown("---")
    
    # 주요 메트릭
    st.markdown("### 📊 주요 메트릭")
    
    # 데이터 로드
    df = load_data()
    
    if df is not None:
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                label="전체 데이터",
                value=f"{len(df):,}건",
                delta="실시간 업데이트"
            )
        
        with col2:
            failure_rate = df['Machine failure'].mean() * 100
            st.metric(
                label="평균 고장률",
                value=f"{failure_rate:.2f}%",
                delta=f"임계값: 5%"
            )
        
        with col3:
            st.metric(
                label="활성 모델",
                value="3개",
                delta="RF, XGB, DL"
            )
        
        with col4:
            st.metric(
                label="최고 정확도",
                value="99.1%",
                delta="Deep Learning"
            )
        
        # 최근 고장 트렌드
        st.markdown("### 📈 최근 고장 트렌드")
        
        # 시뮬레이션 데이터 생성 (실제로는 실시간 데이터 사용)
        dates = pd.date_range(end=datetime.now(), periods=30, freq='D')
        trend_data = pd.DataFrame({
            'Date': dates,
            'Failures': np.random.poisson(3, 30),
            'Predictions': np.random.poisson(3.5, 30)
        })
        
        fig = px.line(trend_data, x='Date', y=['Failures', 'Predictions'],
                     title='일별 고장 발생 트렌드',
                     labels={'value': '건수', 'Date': '날짜'})
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

def page_data_analysis():
    """데이터 분석 페이지"""
    st.title("📊 데이터 분석")
    
    # 데이터 로드
    df = load_data()
    
    if df is None:
        st.error("데이터를 로드할 수 없습니다.")
        return
    
    # 세션 상태에 데이터 저장
    st.session_state.data = df
    st.session_state.data_loaded = True
    
    # 탭 생성
    tab1, tab2, tab3, tab4 = st.tabs(["📈 기본 통계", "🎨 분포", "🔍 상관관계", "⚠️ 고장 분석"])
    
    with tab1:
        st.markdown("### 기본 통계")
        
        # 데이터 정보
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 데이터 정보")
            st.write(f"- 전체 샘플: {len(df):,}개")
            st.write(f"- 특성 개수: {df.shape[1]}개")
            st.write(f"- 고장 샘플: {df['Machine failure'].sum():,}개")
            st.write(f"- 정상 샘플: {(1-df['Machine failure']).sum():,}개")
        
        with col2:
            st.markdown("#### 제품 타입별 분포")
            type_counts = df['Type'].value_counts()
            fig = px.pie(values=type_counts.values, names=type_counts.index,
                        title="제품 타입 분포")
            st.plotly_chart(fig, use_container_width=True)
        
        # 기술 통계
        st.markdown("#### 수치형 변수 통계")
        numeric_cols = ['Air_temperature_K', 'Process_temperature_K', 
                       'Rotational_speed_rpm', 'Torque_Nm', 'Tool_wear_min']
        st.dataframe(df[numeric_cols].describe().round(2))
    
    with tab2:
        st.markdown("### 특성 분포")
        
        # 선택할 특성
        selected_feature = st.selectbox("특성 선택", numeric_cols)
        
        # 히스토그램
        fig = make_subplots(rows=1, cols=2,
                           subplot_titles=['전체 분포', '고장 여부별 분포'])
        
        # 전체 분포
        fig.add_trace(
            go.Histogram(x=df[selected_feature], name='전체',
                        marker_color='lightblue'),
            row=1, col=1
        )
        
        # 고장 여부별 분포
        for failure in [0, 1]:
            data = df[df['Machine failure'] == failure][selected_feature]
            name = '고장' if failure == 1 else '정상'
            color = 'red' if failure == 1 else 'green'
            fig.add_trace(
                go.Histogram(x=data, name=name, marker_color=color,
                           opacity=0.7),
                row=1, col=2
            )
        
        fig.update_layout(height=400, showlegend=True)
        st.plotly_chart(fig, use_container_width=True)
        
        # Box plot
        fig_box = px.box(df, y=selected_feature, x='Machine failure',
                        title=f'{selected_feature} Box Plot',
                        labels={'Machine failure': '고장 여부'})
        st.plotly_chart(fig_box, use_container_width=True)
    
    with tab3:
        st.markdown("### 상관관계 분석")
        
        # 상관관계 행렬
        corr_cols = numeric_cols + ['Machine failure']
        corr_matrix = df[corr_cols].corr()
        
        fig = px.imshow(corr_matrix,
                       labels=dict(x="특성", y="특성", color="상관계수"),
                       title="상관관계 히트맵",
                       color_continuous_scale='RdBu_r',
                       zmin=-1, zmax=1)
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
        
        # Machine failure와의 상관관계
        st.markdown("#### Machine Failure와의 상관관계")
        failure_corr = corr_matrix['Machine failure'].drop('Machine failure').sort_values(ascending=False)
        
        fig_bar = px.bar(x=failure_corr.values, y=failure_corr.index,
                        orientation='h',
                        title="Machine Failure와의 상관계수",
                        labels={'x': '상관계수', 'y': '특성'})
        st.plotly_chart(fig_bar, use_container_width=True)
    
    with tab4:
        st.markdown("### 고장 유형 분석")
        
        # 고장 유형별 발생 건수
        failure_types = ['TWF', 'HDF', 'PWF', 'OSF', 'RNF']
        failure_counts = df[failure_types].sum()
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.bar(x=failure_counts.index, y=failure_counts.values,
                        title="고장 유형별 발생 건수",
                        labels={'x': '고장 유형', 'y': '발생 건수'})
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # 고장 유형 설명
            st.markdown("#### 고장 유형 설명")
            st.write("""
            - **TWF**: Tool Wear Failure (도구 마모)
            - **HDF**: Heat Dissipation Failure (열 방출)
            - **PWF**: Power Failure (전력 고장)
            - **OSF**: Overstrain Failure (과부하)
            - **RNF**: Random Failure (랜덤 고장)
            """)
            
            # 고장 유형별 비율
            total_failures = failure_counts.sum()
            for ft, count in failure_counts.items():
                percentage = (count / total_failures) * 100
                st.metric(ft, f"{count}건", f"{percentage:.1f}%")

def page_prediction():
    """실시간 예측 페이지"""
    st.title("🎯 실시간 예측")
    
    # 모델 체크
    if not st.session_state.model_loaded:
        st.warning("⚠️ 먼저 사이드바에서 모델을 로드하세요.")
        return
    
    st.markdown("### 입력 데이터")
    
    # 입력 폼
    col1, col2, col3 = st.columns(3)
    
    with col1:
        air_temp = st.number_input("공기 온도 [K]", 
                                   min_value=295.0, max_value=305.0, 
                                   value=300.0, step=0.1)
        rotational_speed = st.number_input("회전 속도 [rpm]", 
                                          min_value=1000, max_value=3000, 
                                          value=1500, step=10)
    
    with col2:
        process_temp = st.number_input("공정 온도 [K]", 
                                      min_value=305.0, max_value=315.0, 
                                      value=310.0, step=0.1)
        torque = st.number_input("토크 [Nm]", 
                               min_value=0.0, max_value=80.0, 
                               value=40.0, step=0.5)
    
    with col3:
        tool_wear = st.number_input("도구 마모도 [min]", 
                                   min_value=0, max_value=300, 
                                   value=100, step=5)
        product_type = st.selectbox("제품 타입", ["L", "M", "H"])
    
    # Type 인코딩
    type_encoding = {"H": 0, "L": 1, "M": 2}
    
    # 예측 버튼
    if st.button("🔮 예측 실행", type="primary", use_container_width=True):
        
        # 특성 준비
        features = pd.DataFrame({
            'Air_temperature_K': [air_temp],
            'Process_temperature_K': [process_temp],
            'Rotational_speed_rpm': [rotational_speed],
            'Torque_Nm': [torque],
            'Tool_wear_min': [tool_wear],
            'Type_encoded': [type_encoding[product_type]]
        })
        
        st.markdown("---")
        st.markdown("### 예측 결과")
        
        # 각 모델별 예측
        results = []
        cols = st.columns(len(st.session_state.models))
        
        for idx, (model_name, model) in enumerate(st.session_state.models.items()):
            with cols[idx]:
                st.markdown(f"#### {model_name}")
                
                # 예측
                prediction, probability = get_prediction(
                    model, model_name, features, st.session_state.scaler
                )
                
                if prediction is not None:
                    # 결과 표시
                    if prediction == 0:
                        st.success("✅ 정상")
                        color = "green"
                    else:
                        st.error("⚠️ 고장 예측")
                        color = "red"
                    
                    # 확률 표시
                    st.metric("예측 확률", f"{probability:.2%}")
                    
                    # 게이지 차트
                    fig = go.Figure(go.Indicator(
                        mode = "gauge+number",
                        value = probability * 100,
                        title = {'text': "고장 확률 (%)"},
                        gauge = {
                            'axis': {'range': [None, 100]},
                            'bar': {'color': color},
                            'steps': [
                                {'range': [0, 30], 'color': "lightgreen"},
                                {'range': [30, 70], 'color': "yellow"},
                                {'range': [70, 100], 'color': "lightcoral"}
                            ],
                            'threshold': {
                                'line': {'color': "red", 'width': 4},
                                'thickness': 0.75,
                                'value': 50
                            }
                        }
                    ))
                    fig.update_layout(height=250)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    results.append({
                        'Model': model_name,
                        'Prediction': '고장' if prediction == 1 else '정상',
                        'Probability': probability
                    })
        
        # AI Agent 분석
        if st.session_state.ai_agent:
            st.markdown("---")
            st.markdown("### 🤖 AI 분석")
            
            with st.spinner("AI가 분석 중입니다..."):
                # 가장 높은 확률의 모델 선택
                best_result = max(results, key=lambda x: x['Probability'])
                
                # AI 분석 요청
                analysis = st.session_state.ai_agent.interpret_prediction(
                    model_name=best_result['Model'],
                    prediction=1 if best_result['Prediction'] == '고장' else 0,
                    probability=best_result['Probability'],
                    input_features={
                        'Air_temperature_K': air_temp,
                        'Process_temperature_K': process_temp,
                        'Rotational_speed_rpm': rotational_speed,
                        'Torque_Nm': torque,
                        'Tool_wear_min': tool_wear,
                        'Type': product_type
                    }
                )
                
                st.info(analysis.explanation)

def page_model_comparison():
    """모델 비교 페이지"""
    st.title("📈 모델 성능 비교")
    
    # 모델 성능 데이터 (하드코딩 - 실제로는 저장된 메트릭 사용)
    performance_data = pd.DataFrame({
        'Model': ['Random Forest', 'XGBoost', 'Deep Learning'],
        'Accuracy': [0.9825, 0.9885, 0.9910],
        'Precision': [0.9024, 0.9245, 0.9500],
        'Recall': [0.5441, 0.7206, 0.7800],
        'F1-Score': [0.6789, 0.8099, 0.8565]
    })
    
    # 메트릭 선택
    metrics = st.multiselect(
        "비교할 메트릭 선택",
        ['Accuracy', 'Precision', 'Recall', 'F1-Score'],
        default=['Accuracy', 'F1-Score']
    )
    
    if metrics:
        # 막대 그래프
        fig = go.Figure()
        
        for metric in metrics:
            fig.add_trace(go.Bar(
                name=metric,
                x=performance_data['Model'],
                y=performance_data[metric],
                text=performance_data[metric].round(4),
                textposition='auto',
            ))
        
        fig.update_layout(
            title="모델별 성능 비교",
            xaxis_title="모델",
            yaxis_title="점수",
            barmode='group',
            yaxis_range=[0, 1],
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # 레이더 차트
        st.markdown("### 레이더 차트")
        
        fig_radar = go.Figure()
        
        for _, row in performance_data.iterrows():
            fig_radar.add_trace(go.Scatterpolar(
                r=[row['Accuracy'], row['Precision'], row['Recall'], row['F1-Score']],
                theta=['Accuracy', 'Precision', 'Recall', 'F1-Score'],
                fill='toself',
                name=row['Model']
            ))
        
        fig_radar.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )),
            showlegend=True,
            height=500
        )
        
        st.plotly_chart(fig_radar, use_container_width=True)
        
        # 성능 테이블
        st.markdown("### 상세 성능 메트릭")
        st.dataframe(
            performance_data.style.highlight_max(axis=0, subset=metrics),
            use_container_width=True
        )
        
        # 최고 모델
        best_model_by_metric = {}
        for metric in metrics:
            best_model = performance_data.loc[performance_data[metric].idxmax(), 'Model']
            best_value = performance_data[metric].max()
            best_model_by_metric[metric] = f"{best_model} ({best_value:.4f})"
        
        st.markdown("### 🏆 메트릭별 최고 성능 모델")
        for metric, model_info in best_model_by_metric.items():
            st.write(f"- **{metric}**: {model_info}")
    
    # Feature Importance 비교
    st.markdown("---")
    st.markdown("### Feature Importance 비교")
    
    # 예시 데이터
    importance_data = pd.DataFrame({
        'Feature': ['Torque_Nm', 'Rotational_speed_rpm', 'Tool_wear_min', 
                    'Air_temperature_K', 'Process_temperature_K', 'Type_encoded'],
        'Random Forest': [0.3705, 0.2359, 0.1569, 0.1059, 0.1130, 0.0179],
        'XGBoost': [0.2958, 0.1780, 0.1878, 0.1580, 0.0981, 0.0822]
    })
    
    fig_imp = go.Figure()
    
    for model in ['Random Forest', 'XGBoost']:
        fig_imp.add_trace(go.Bar(
            name=model,
            x=importance_data['Feature'],
            y=importance_data[model],
            text=importance_data[model].round(3),
            textposition='auto',
        ))
    
    fig_imp.update_layout(
        title="Feature Importance 비교",
        xaxis_title="특성",
        yaxis_title="중요도",
        barmode='group',
        height=400
    )
    
    st.plotly_chart(fig_imp, use_container_width=True)

def page_ai_assistant():
    """AI Assistant 페이지 - 대화형 분석 기능 강화"""
    st.title("🤖 AI Assistant - 대화형 분석")
    
    # AI Agent 초기화
    if st.session_state.ai_agent is None:
        st.warning("⚠️ OpenAI API 키를 입력하여 AI Agent를 활성화하세요")
        api_key = st.text_input("OpenAI API Key", type="password", 
                               help="API 키는 https://platform.openai.com/api-keys 에서 발급받을 수 있습니다")
        if st.button("AI Agent 초기화", type="primary"):
            try:
                st.session_state.ai_agent = ManufacturingAIAgent(api_key)
                st.success("✅ AI Agent 초기화 완료!")
                st.rerun()
            except Exception as e:
                st.error(f"초기화 실패: {e}")
                return
    
    if st.session_state.ai_agent:
        # 탭 생성
        tab1, tab2, tab3, tab4 = st.tabs([
            "💬 자연어 질의응답", 
            "🔮 What-if 시나리오", 
            "📊 비교 분석",
            "📜 대화 기록"
        ])
        
        with tab1:
            st.markdown("### 💬 데이터에 대해 자연어로 질문하세요")
            st.info("예: '평균 온도는 얼마야?', '고장이 가장 많이 발생한 시간대는?', '토크와 고장률의 관계는?'")
            
            # 예제 질문 카테고리별로 정리
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**📈 통계 질문 예시**")
                stat_questions = [
                    "평균 온도와 회전속도는?",
                    "도구 마모도의 최대값은?",
                    "전체 데이터의 통계 요약"
                ]
                for q in stat_questions:
                    if st.button(q, key=f"stat_{q}"):
                        st.session_state.current_question = q
            
            with col2:
                st.markdown("**⚠️ 고장 분석 예시**")
                failure_questions = [
                    "고장률이 얼마나 되나요?",
                    "어떤 고장 유형이 가장 많아?",
                    "고장 시 평균 온도는?"
                ]
                for q in failure_questions:
                    if st.button(q, key=f"fail_{q}"):
                        st.session_state.current_question = q
            
            # 질문 입력
            user_query = st.text_area(
                "질문을 입력하세요",
                value=st.session_state.get('current_question', ''),
                height=100,
                placeholder="데이터에 대해 궁금한 점을 자유롭게 질문하세요..."
            )
            
            if st.button("🔍 분석하기", type="primary"):
                if user_query and st.session_state.data_loaded:
                    with st.spinner("AI가 데이터를 분석 중입니다..."):
                        # 자연어 쿼리 분석
                        result = st.session_state.ai_agent.analyze_natural_query(
                            query=user_query,
                            df=st.session_state.data
                        )
                        
                        # 결과 타입에 따른 표시
                        st.markdown("### 📊 분석 결과")
                        
                        if result['type'] == 'statistics':
                            # 통계 결과 표시
                            if 'statistics' in result['result']:
                                st.markdown("**📈 데이터 통계**")
                                stats_df = pd.DataFrame(result['result']['statistics']).T
                                st.dataframe(stats_df, use_container_width=True)
                                
                                # 시각화
                                fig = go.Figure()
                                for col in stats_df.columns:
                                    if col == '평균':
                                        fig.add_trace(go.Bar(
                                            name=col,
                                            x=stats_df.index,
                                            y=stats_df[col],
                                            text=stats_df[col].round(2),
                                            textposition='auto'
                                        ))
                                
                                fig.update_layout(
                                    title="주요 변수 평균값",
                                    xaxis_title="변수",
                                    yaxis_title="값",
                                    height=400
                                )
                                st.plotly_chart(fig, use_container_width=True)
                            
                            st.info(result['result'].get('explanation', ''))
                        
                        elif result['type'] == 'failure_analysis':
                            # 고장 분석 결과 표시
                            if 'failure_stats' in result['result']:
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    st.metric("총 고장 수", result['result']['failure_stats'].get('총 고장 수', 0))
                                with col2:
                                    st.metric("고장률", result['result']['failure_stats'].get('고장률', '0%'))
                                with col3:
                                    st.metric("데이터 수", len(st.session_state.data))
                                
                                # 고장 유형별 차트
                                if '고장 유형별' in result['result']['failure_stats']:
                                    failure_types = result['result']['failure_stats']['고장 유형별']
                                    if failure_types:
                                        fig = go.Figure(data=[go.Pie(
                                            labels=list(failure_types.keys()),
                                            values=list(failure_types.values()),
                                            hole=.3
                                        )])
                                        fig.update_layout(title="고장 유형 분포")
                                        st.plotly_chart(fig, use_container_width=True)
                            
                            st.info(result['result'].get('explanation', ''))
                        
                        elif result['type'] == 'trend_analysis':
                            # 트렌드 분석 결과 표시
                            if 'trends' in result['result']:
                                st.markdown("**📉 트렌드 분석**")
                                trends_df = pd.DataFrame(result['result']['trends']).T
                                st.dataframe(trends_df, use_container_width=True)
                            
                            st.info(result['result'].get('explanation', ''))
                        
                        else:
                            # 일반 응답
                            st.info(result['result'])
                        
                        # 대화 기록 저장
                        st.session_state.chat_history.append({
                            "user": user_query,
                            "assistant": result.get('result', {}).get('explanation', result.get('result', '')),
                            "timestamp": datetime.now(),
                            "type": result['type']
                        })
                        st.session_state.current_question = ""
                
                elif not st.session_state.data_loaded:
                    st.warning("먼저 데이터를 로드해주세요 (데이터 탐색 페이지)")
        
        with tab2:
            st.markdown("### 🔮 What-if 시나리오 분석")
            st.info("조건을 변경했을 때의 영향을 예측합니다")
            
            if st.session_state.data_loaded:
                # 현재 평균값 계산
                current_values = {
                    'Air_temperature_K': st.session_state.data['Air_temperature_K'].mean(),
                    'Process_temperature_K': st.session_state.data['Process_temperature_K'].mean(),
                    'Rotational_speed_rpm': st.session_state.data['Rotational_speed_rpm'].mean(),
                    'Torque_Nm': st.session_state.data['Torque_Nm'].mean(),
                    'Tool_wear_min': st.session_state.data['Tool_wear_min'].mean()
                }
                
                # 시나리오 선택
                scenario_type = st.selectbox(
                    "시나리오 유형",
                    ["온도 변화", "회전속도 변화", "토크 변화", "복합 시나리오", "사용자 정의"]
                )
                
                scenario_text = ""
                if scenario_type == "온도 변화":
                    temp_change = st.slider("온도 변화 (K)", -20, 20, 0)
                    scenario_text = f"온도가 {temp_change}도 {'증가' if temp_change > 0 else '감소'}하면 어떻게 될까?"
                elif scenario_type == "회전속도 변화":
                    speed_change = st.slider("회전속도 변화 (%)", -50, 50, 0)
                    scenario_text = f"회전속도가 {abs(speed_change)}% {'증가' if speed_change > 0 else '감소'}하면?"
                elif scenario_type == "토크 변화":
                    torque_change = st.slider("토크 변화 (%)", -50, 50, 0)
                    scenario_text = f"토크가 {abs(torque_change)}% {'증가' if torque_change > 0 else '감소'}하면?"
                elif scenario_type == "복합 시나리오":
                    st.markdown("**복합 조건 설정**")
                    col1, col2 = st.columns(2)
                    with col1:
                        temp_change = st.number_input("온도 변화 (K)", -20, 20, 0)
                        speed_change = st.number_input("회전속도 변화 (%)", -50, 50, 0)
                    with col2:
                        torque_change = st.number_input("토크 변화 (%)", -50, 50, 0)
                        wear_change = st.number_input("마모도 변화 (분)", -100, 100, 0)
                    scenario_text = "복합 시나리오: 여러 조건이 동시에 변화"
                else:
                    scenario_text = st.text_input("시나리오를 자유롭게 설명하세요", 
                                                 "예: 여름철 온도가 10도 상승하고 생산량을 20% 늘리면?")
                
                # 현재 값 표시
                st.markdown("**현재 평균값**")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("온도", f"{current_values['Air_temperature_K']:.1f} K")
                    st.metric("공정 온도", f"{current_values['Process_temperature_K']:.1f} K")
                with col2:
                    st.metric("회전속도", f"{current_values['Rotational_speed_rpm']:.0f} rpm")
                    st.metric("토크", f"{current_values['Torque_Nm']:.1f} Nm")
                with col3:
                    st.metric("도구 마모도", f"{current_values['Tool_wear_min']:.0f} min")
                
                if st.button("🔮 시나리오 분석 실행", type="primary"):
                    with st.spinner("시나리오를 분석 중입니다..."):
                        result = st.session_state.ai_agent.what_if_scenario(
                            scenario=scenario_text,
                            current_values=current_values,
                            models=st.session_state.models if st.session_state.model_loaded else None
                        )
                        
                        st.markdown("### 📊 시나리오 분석 결과")
                        
                        # 값 변화 표시
                        if 'impact_analysis' in result:
                            st.markdown("**📈 변경된 값**")
                            impact_df = pd.DataFrame(result['impact_analysis']).T
                            if not impact_df.empty:
                                st.dataframe(impact_df, use_container_width=True)
                        
                        # 위험도 변화
                        if 'prediction_changes' in result:
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("기존 위험도", result['prediction_changes'].get('기존 위험도', 'N/A'))
                            with col2:
                                st.metric("변경 후 위험도", result['prediction_changes'].get('변경 후 위험도', 'N/A'))
                            with col3:
                                st.metric("위험도 변화", result['prediction_changes'].get('위험도 변화', 'N/A'))
                        
                        # AI 설명
                        st.markdown("**🤖 AI 분석**")
                        st.info(result.get('explanation', ''))
            else:
                st.warning("먼저 데이터를 로드해주세요 (데이터 탐색 페이지)")
        
        with tab3:
            st.markdown("### 📊 기간/모델 비교 분석")
            
            comparison_type = st.radio("비교 유형", ["모델 성능 비교", "기간별 데이터 비교"])
            
            if comparison_type == "모델 성능 비교":
                if st.button("🔍 모델 성능 비교 분석", type="primary"):
                    with st.spinner("모델을 비교 분석 중..."):
                        comparison = st.session_state.ai_agent.compare_models()
                        st.markdown("### 🏆 모델 성능 비교 결과")
                        st.info(comparison)
            
            else:  # 기간별 데이터 비교
                if st.session_state.data_loaded:
                    st.markdown("**데이터를 두 기간으로 분할하여 비교합니다**")
                    
                    split_point = st.slider(
                        "분할 지점 (%)",
                        10, 90, 50,
                        help="전체 데이터를 어느 지점에서 분할할지 선택"
                    )
                    
                    split_idx = int(len(st.session_state.data) * split_point / 100)
                    df1 = st.session_state.data.iloc[:split_idx]
                    df2 = st.session_state.data.iloc[split_idx:]
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("기간 1 데이터", f"{len(df1):,} 건")
                    with col2:
                        st.metric("기간 2 데이터", f"{len(df2):,} 건")
                    
                    if st.button("📊 기간별 비교 분석", type="primary"):
                        with st.spinner("기간을 비교 분석 중..."):
                            result = st.session_state.ai_agent.compare_periods(
                                df1=df1,
                                df2=df2,
                                period1_name="전반기",
                                period2_name="후반기"
                            )
                            
                            st.markdown("### 📈 기간별 비교 결과")
                            
                            # 비교 테이블
                            if 'comparison' in result:
                                comparison_df = pd.DataFrame(result['comparison'])
                                st.dataframe(comparison_df, use_container_width=True)
                                
                                # 변화 시각화
                                metrics = ['Air_temperature_K', 'Process_temperature_K', 'Rotational_speed_rpm', 'Torque_Nm']
                                fig = go.Figure()
                                
                                for period in ['전반기', '후반기']:
                                    values = [result['comparison'][period].get(m, 0) for m in metrics]
                                    fig.add_trace(go.Bar(name=period, x=metrics, y=values))
                                
                                fig.update_layout(
                                    title="기간별 주요 지표 비교",
                                    xaxis_title="지표",
                                    yaxis_title="값",
                                    barmode='group'
                                )
                                st.plotly_chart(fig, use_container_width=True)
                            
                            # AI 설명
                            st.markdown("**🤖 AI 분석**")
                            st.info(result.get('explanation', ''))
                else:
                    st.warning("먼저 데이터를 로드해주세요 (데이터 탐색 페이지)")
        
        with tab4:
            st.markdown("### 📜 대화 기록")
            
            if st.session_state.chat_history:
                # 필터링 옵션
                filter_type = st.selectbox("대화 유형 필터", 
                                          ["전체", "통계", "고장 분석", "트렌드", "일반"])
                
                # 대화 기록 표시
                for i, chat in enumerate(reversed(st.session_state.chat_history[-10:])):
                    chat_type = chat.get('type', 'general')
                    
                    if filter_type == "전체" or \
                       (filter_type == "통계" and chat_type == "statistics") or \
                       (filter_type == "고장 분석" and chat_type == "failure_analysis") or \
                       (filter_type == "트렌드" and chat_type == "trend_analysis") or \
                       (filter_type == "일반" and chat_type == "general"):
                        
                        with st.expander(f"🕐 {chat['timestamp'].strftime('%Y-%m-%d %H:%M:%S')} - {chat['user'][:50]}..."):
                            st.markdown(f"**👤 질문**: {chat['user']}")
                            st.markdown(f"**🤖 답변**: {chat['assistant']}")
                            st.caption(f"유형: {chat_type}")
                
                # 대화 기록 다운로드
                if st.button("💾 대화 기록 다운로드"):
                    history_json = json.dumps(
                        [{"timestamp": c['timestamp'].isoformat(), 
                          "user": c['user'], 
                          "assistant": c['assistant'],
                          "type": c.get('type', 'general')} 
                         for c in st.session_state.chat_history],
                        indent=2,
                        ensure_ascii=False
                    )
                    st.download_button(
                        label="📥 JSON 다운로드",
                        data=history_json,
                        file_name=f"chat_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        mime="application/json"
                    )
            else:
                st.info("아직 대화 기록이 없습니다. 질문을 시작해보세요!")

def page_learning():
    """학습 자료 페이지"""
    st.title("📚 학습 자료")
    
    st.markdown("""
    ### 4일차 프로젝트 학습 내용
    
    #### 1. Streamlit 기초
    - 웹 애플리케이션 구조
    - 위젯과 레이아웃
    - 세션 상태 관리
    - 캐싱과 성능 최적화
    
    #### 2. AI Agent 개발
    - OpenAI API 활용
    - 프롬프트 엔지니어링
    - 컨텍스트 관리
    - 응답 처리
    
    #### 3. 데이터 시각화
    - Plotly를 활용한 인터랙티브 차트
    - 실시간 데이터 업데이트
    - 대시보드 디자인
    
    #### 4. 모델 통합
    - 다중 모델 관리
    - 예측 결과 비교
    - 성능 메트릭 시각화
    """)
    
    # 코드 예제
    with st.expander("📝 Streamlit 기본 예제"):
        st.code("""
        import streamlit as st
        
        # 제목
        st.title("My Dashboard")
        
        # 입력 위젯
        name = st.text_input("이름")
        age = st.slider("나이", 0, 100, 25)
        
        # 버튼
        if st.button("제출"):
            st.write(f"안녕하세요, {name}님 ({age}세)")
        
        # 데이터프레임
        df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
        st.dataframe(df)
        
        # 차트
        st.line_chart(df)
        """, language='python')
    
    with st.expander("📝 AI Agent 호출 예제"):
        st.code("""
        from ai_agent import ManufacturingAIAgent
        
        # Agent 초기화
        agent = ManufacturingAIAgent(api_key="your-key")
        
        # 예측 해석
        result = agent.interpret_prediction(
            model_name="XGBoost",
            prediction=1,
            probability=0.85,
            input_features={...}
        )
        
        # 질의응답
        answer = agent.ask_question("고장을 예방하는 방법은?")
        """, language='python')
    
    # 참고 자료
    st.markdown("""
    ### 📖 참고 자료
    
    - [Streamlit Documentation](https://docs.streamlit.io/)
    - [OpenAI API Documentation](https://platform.openai.com/docs)
    - [Plotly Python Documentation](https://plotly.com/python/)
    - [Manufacturing Dataset Paper](https://doi.org/10.1109/AI4I49448.2020.00023)
    """)
    
    # 실습 과제
    st.markdown("""
    ### 💡 실습 과제
    
    1. **대시보드 커스터마이징**
       - 새로운 차트 유형 추가
       - 테마 변경
       - 추가 메트릭 구현
    
    2. **AI Agent 확장**
       - 커스텀 프롬프트 작성
       - 다국어 지원 추가
       - 음성 인터페이스 구현
    
    3. **실시간 모니터링**
       - 웹소켓 연결
       - 알림 시스템
       - 로그 분석
    """)

# =============================================================================
# 7. 메인 앱
# =============================================================================

def main():
    """메인 애플리케이션"""
    
    # 사이드바 렌더링
    render_sidebar()
    
    # 페이지 라우팅
    if st.session_state.page == "Home":
        page_home()
    elif st.session_state.page == "데이터":
        page_data_analysis()
    elif st.session_state.page == "실시간":
        page_prediction()
    elif st.session_state.page == "모델":
        page_model_comparison()
    elif st.session_state.page == "AI":
        page_ai_assistant()
    elif st.session_state.page == "학습":
        page_learning()
    
    # 푸터
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: gray;'>
        <p>© 2024 Manufacturing Predictive Maintenance System | 
        Developed for S-OIL AI Training Program</p>
    </div>
    """, unsafe_allow_html=True)

# =============================================================================
# 8. 앱 실행
# =============================================================================

if __name__ == "__main__":
    main()

# =============================================================================
# 실행 방법
# =============================================================================
"""
터미널에서 실행:
$ streamlit run 04_dashboard.py

옵션:
$ streamlit run 04_dashboard.py --server.port 8501 --server.address localhost

배포:
1. Streamlit Cloud: https://streamlit.io/cloud
2. Heroku: Procfile 생성 필요
3. AWS/GCP: Docker 컨테이너화 권장
"""