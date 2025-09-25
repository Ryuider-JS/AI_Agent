"""
유틸리티 함수 모음
=====================================

이 모듈은 4일차 프로젝트에서 공통적으로 사용되는
유틸리티 함수들을 포함합니다.

Author: Utility Development Team
Date: 2024
"""

import pandas as pd
import numpy as np
import pickle
import json
import os
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime, timedelta
import logging

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =============================================================================
# 1. 데이터 처리 함수
# =============================================================================

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    데이터 전처리 함수
    
    Args:
        df: 원본 데이터프레임
    
    Returns:
        전처리된 데이터프레임
    """
    # 컬럼명 변경 (XGBoost 호환)
    column_rename_map = {
        "Air temperature [K]": "Air_temperature_K",
        "Process temperature [K]": "Process_temperature_K",
        "Rotational speed [rpm]": "Rotational_speed_rpm",
        "Torque [Nm]": "Torque_Nm",
        "Tool wear [min]": "Tool_wear_min",
    }
    
    df_processed = df.copy()
    df_processed.rename(columns=column_rename_map, inplace=True)
    
    # Type 인코딩
    if 'Type' in df_processed.columns:
        type_encoding = {"H": 0, "L": 1, "M": 2}
        df_processed['Type_encoded'] = df_processed['Type'].map(type_encoding)
    
    return df_processed

def create_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """
    Feature Engineering 수행
    
    Args:
        df: 입력 데이터프레임
    
    Returns:
        새로운 특성이 추가된 데이터프레임
    """
    df_fe = df.copy()
    
    # 온도 차이
    if 'Process_temperature_K' in df.columns and 'Air_temperature_K' in df.columns:
        df_fe['Temperature_diff'] = df_fe['Process_temperature_K'] - df_fe['Air_temperature_K']
    
    # 파워 (회전속도 × 토크)
    if 'Rotational_speed_rpm' in df.columns and 'Torque_Nm' in df.columns:
        df_fe['Power'] = (df_fe['Rotational_speed_rpm'] * df_fe['Torque_Nm']) / 9550
    
    # 도구 마모 레벨 (범주화)
    if 'Tool_wear_min' in df.columns:
        df_fe['Tool_wear_level'] = pd.cut(df_fe['Tool_wear_min'], 
                                          bins=[0, 50, 150, 250, 300],
                                          labels=['Low', 'Medium', 'High', 'Critical'])
    
    return df_fe

# =============================================================================
# 2. 모델 관련 함수
# =============================================================================

def save_model(model: Any, filepath: str) -> bool:
    """
    모델 저장
    
    Args:
        model: 저장할 모델 객체
        filepath: 저장 경로
    
    Returns:
        성공 여부
    """
    try:
        # 디렉토리 생성
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # 모델 저장
        with open(filepath, 'wb') as f:
            pickle.dump(model, f)
        
        logger.info(f"모델 저장 완료: {filepath}")
        return True
    
    except Exception as e:
        logger.error(f"모델 저장 실패: {e}")
        return False

def load_model(filepath: str) -> Optional[Any]:
    """
    모델 로드
    
    Args:
        filepath: 모델 파일 경로
    
    Returns:
        로드된 모델 또는 None
    """
    try:
        with open(filepath, 'rb') as f:
            model = pickle.load(f)
        logger.info(f"모델 로드 완료: {filepath}")
        return model
    
    except Exception as e:
        logger.error(f"모델 로드 실패: {e}")
        return None

def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    평가 메트릭 계산
    
    Args:
        y_true: 실제 값
        y_pred: 예측 값
    
    Returns:
        메트릭 딕셔너리
    """
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1_score': f1_score(y_true, y_pred, zero_division=0)
    }
    
    return metrics

# =============================================================================
# 3. 시각화 헬퍼 함수
# =============================================================================

def get_color_palette() -> Dict[str, str]:
    """
    일관된 색상 팔레트 반환
    
    Returns:
        색상 딕셔너리
    """
    return {
        'primary': '#1f77b4',
        'success': '#2ca02c',
        'warning': '#ff7f0e',
        'danger': '#d62728',
        'info': '#17a2b8',
        'light': '#f8f9fa',
        'dark': '#343a40'
    }

def format_percentage(value: float, decimals: int = 2) -> str:
    """
    백분율 포맷팅
    
    Args:
        value: 값 (0-1 사이)
        decimals: 소수점 자리수
    
    Returns:
        포맷된 문자열
    """
    return f"{value * 100:.{decimals}f}%"

def format_number(value: float, decimals: int = 2, thousands_sep: bool = True) -> str:
    """
    숫자 포맷팅
    
    Args:
        value: 숫자 값
        decimals: 소수점 자리수
        thousands_sep: 천단위 구분자 사용 여부
    
    Returns:
        포맷된 문자열
    """
    if thousands_sep:
        return f"{value:,.{decimals}f}"
    else:
        return f"{value:.{decimals}f}"

# =============================================================================
# 4. 파일 처리 함수
# =============================================================================

def save_json(data: Dict, filepath: str) -> bool:
    """
    JSON 파일 저장
    
    Args:
        data: 저장할 데이터
        filepath: 파일 경로
    
    Returns:
        성공 여부
    """
    try:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logger.info(f"JSON 저장 완료: {filepath}")
        return True
    
    except Exception as e:
        logger.error(f"JSON 저장 실패: {e}")
        return False

def load_json(filepath: str) -> Optional[Dict]:
    """
    JSON 파일 로드
    
    Args:
        filepath: 파일 경로
    
    Returns:
        로드된 데이터 또는 None
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        logger.info(f"JSON 로드 완료: {filepath}")
        return data
    
    except Exception as e:
        logger.error(f"JSON 로드 실패: {e}")
        return None

# =============================================================================
# 5. 예측 관련 함수
# =============================================================================

def prepare_prediction_input(
    air_temp: float,
    process_temp: float,
    rotational_speed: int,
    torque: float,
    tool_wear: int,
    product_type: str
) -> pd.DataFrame:
    """
    예측을 위한 입력 데이터 준비
    
    Args:
        air_temp: 공기 온도
        process_temp: 공정 온도
        rotational_speed: 회전 속도
        torque: 토크
        tool_wear: 도구 마모도
        product_type: 제품 타입
    
    Returns:
        준비된 데이터프레임
    """
    # Type 인코딩
    type_encoding = {"H": 0, "L": 1, "M": 2}
    
    # DataFrame 생성
    data = pd.DataFrame({
        'Air_temperature_K': [air_temp],
        'Process_temperature_K': [process_temp],
        'Rotational_speed_rpm': [rotational_speed],
        'Torque_Nm': [torque],
        'Tool_wear_min': [tool_wear],
        'Type_encoded': [type_encoding.get(product_type, 0)]
    })
    
    return data

def interpret_prediction_result(
    prediction: int,
    probability: float,
    threshold: float = 0.5
) -> Dict[str, Any]:
    """
    예측 결과 해석
    
    Args:
        prediction: 예측값 (0 또는 1)
        probability: 예측 확률
        threshold: 임계값
    
    Returns:
        해석 결과
    """
    # 신뢰도 계산
    if probability > 0.9:
        confidence = "Very High"
        confidence_level = 5
    elif probability > 0.7:
        confidence = "High"
        confidence_level = 4
    elif probability > 0.5:
        confidence = "Medium"
        confidence_level = 3
    elif probability > 0.3:
        confidence = "Low"
        confidence_level = 2
    else:
        confidence = "Very Low"
        confidence_level = 1
    
    # 위험 수준
    if prediction == 1:
        if probability > 0.8:
            risk_level = "Critical"
            action = "Immediate maintenance required"
        elif probability > 0.6:
            risk_level = "High"
            action = "Schedule maintenance soon"
        else:
            risk_level = "Medium"
            action = "Monitor closely"
    else:
        if probability < 0.2:
            risk_level = "Low"
            action = "Normal operation"
        else:
            risk_level = "Medium-Low"
            action = "Continue monitoring"
    
    return {
        'prediction': 'Failure' if prediction == 1 else 'Normal',
        'probability': probability,
        'confidence': confidence,
        'confidence_level': confidence_level,
        'risk_level': risk_level,
        'recommended_action': action
    }

# =============================================================================
# 6. 시간 관련 함수
# =============================================================================

def generate_time_series_data(
    start_date: datetime,
    end_date: datetime,
    freq: str = 'H'
) -> pd.DatetimeIndex:
    """
    시계열 데이터 생성
    
    Args:
        start_date: 시작 날짜
        end_date: 종료 날짜
        freq: 빈도 ('H', 'D', 'W', 'M')
    
    Returns:
        날짜 인덱스
    """
    return pd.date_range(start=start_date, end=end_date, freq=freq)

def calculate_maintenance_schedule(
    last_maintenance: datetime,
    failure_probability: float,
    tool_wear: int
) -> Dict[str, Any]:
    """
    유지보수 일정 계산
    
    Args:
        last_maintenance: 마지막 유지보수 날짜
        failure_probability: 고장 확률
        tool_wear: 도구 마모도
    
    Returns:
        유지보수 일정 정보
    """
    # 기본 주기 (일)
    base_interval = 30
    
    # 위험도에 따른 조정
    if failure_probability > 0.7:
        adjustment = 0.3  # 70% 단축
    elif failure_probability > 0.5:
        adjustment = 0.5  # 50% 단축
    elif failure_probability > 0.3:
        adjustment = 0.7  # 30% 단축
    else:
        adjustment = 1.0  # 정상
    
    # 도구 마모도에 따른 추가 조정
    if tool_wear > 200:
        wear_adjustment = 0.5
    elif tool_wear > 150:
        wear_adjustment = 0.7
    elif tool_wear > 100:
        wear_adjustment = 0.85
    else:
        wear_adjustment = 1.0
    
    # 최종 주기 계산
    final_interval = int(base_interval * adjustment * wear_adjustment)
    final_interval = max(final_interval, 1)  # 최소 1일
    
    # 다음 유지보수 날짜
    next_maintenance = last_maintenance + timedelta(days=final_interval)
    
    # 긴급도
    days_until = (next_maintenance - datetime.now()).days
    if days_until <= 0:
        urgency = "Overdue"
    elif days_until <= 3:
        urgency = "Urgent"
    elif days_until <= 7:
        urgency = "High"
    elif days_until <= 14:
        urgency = "Medium"
    else:
        urgency = "Low"
    
    return {
        'last_maintenance': last_maintenance,
        'next_maintenance': next_maintenance,
        'days_until': days_until,
        'urgency': urgency,
        'interval_days': final_interval
    }

# =============================================================================
# 7. 검증 함수
# =============================================================================

def validate_input_features(features: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """
    입력 특성 검증
    
    Args:
        features: 입력 특성 딕셔너리
    
    Returns:
        (검증 성공 여부, 오류 메시지 리스트)
    """
    errors = []
    
    # 필수 특성 확인
    required_features = [
        'Air_temperature_K', 'Process_temperature_K', 
        'Rotational_speed_rpm', 'Torque_Nm', 'Tool_wear_min'
    ]
    
    for feat in required_features:
        if feat not in features:
            errors.append(f"Missing required feature: {feat}")
    
    # 값 범위 검증
    if 'Air_temperature_K' in features:
        if not (295 <= features['Air_temperature_K'] <= 305):
            errors.append("Air temperature out of range (295-305 K)")
    
    if 'Process_temperature_K' in features:
        if not (305 <= features['Process_temperature_K'] <= 315):
            errors.append("Process temperature out of range (305-315 K)")
    
    if 'Rotational_speed_rpm' in features:
        if not (1000 <= features['Rotational_speed_rpm'] <= 3000):
            errors.append("Rotational speed out of range (1000-3000 rpm)")
    
    if 'Torque_Nm' in features:
        if not (0 <= features['Torque_Nm'] <= 100):
            errors.append("Torque out of range (0-100 Nm)")
    
    if 'Tool_wear_min' in features:
        if not (0 <= features['Tool_wear_min'] <= 300):
            errors.append("Tool wear out of range (0-300 min)")
    
    return len(errors) == 0, errors

# =============================================================================
# 8. 보고서 생성 함수
# =============================================================================

def generate_analysis_report(
    data: pd.DataFrame,
    predictions: Dict[str, Any],
    timestamp: datetime = None
) -> str:
    """
    분석 보고서 생성
    
    Args:
        data: 분석 데이터
        predictions: 예측 결과
        timestamp: 보고서 생성 시간
    
    Returns:
        보고서 문자열
    """
    if timestamp is None:
        timestamp = datetime.now()
    
    report = f"""
    =====================================
    제조업 예측 유지보수 분석 보고서
    =====================================
    
    생성 시간: {timestamp.strftime('%Y-%m-%d %H:%M:%S')}
    
    1. 데이터 개요
    --------------
    - 전체 샘플 수: {len(data):,}
    - 고장 건수: {data['Machine failure'].sum() if 'Machine failure' in data.columns else 'N/A'}
    - 고장률: {data['Machine failure'].mean() * 100:.2f}% if 'Machine failure' in data.columns else 'N/A'
    
    2. 예측 결과
    ------------
    """
    
    for model_name, result in predictions.items():
        report += f"""
    {model_name}:
    - 예측: {result.get('prediction', 'N/A')}
    - 확률: {result.get('probability', 0) * 100:.2f}%
    - 신뢰도: {result.get('confidence', 'N/A')}
    """
    
    report += """
    
    3. 권장 조치
    ------------
    Based on the analysis, the following actions are recommended:
    - Immediate: Check critical components
    - Short-term: Schedule maintenance
    - Long-term: Update maintenance strategy
    
    =====================================
    End of Report
    =====================================
    """
    
    return report

# =============================================================================
# 9. 테스트 함수
# =============================================================================

def run_tests():
    """유틸리티 함수 테스트"""
    print("=" * 60)
    print("유틸리티 함수 테스트")
    print("=" * 60)
    
    # 1. 데이터 검증 테스트
    print("\n1. 입력 검증 테스트")
    test_features = {
        'Air_temperature_K': 300,
        'Process_temperature_K': 310,
        'Rotational_speed_rpm': 1500,
        'Torque_Nm': 40,
        'Tool_wear_min': 100
    }
    
    is_valid, errors = validate_input_features(test_features)
    print(f"검증 결과: {'성공' if is_valid else '실패'}")
    if errors:
        print(f"오류: {errors}")
    
    # 2. 예측 해석 테스트
    print("\n2. 예측 해석 테스트")
    result = interpret_prediction_result(1, 0.85)
    print(f"예측 결과: {result}")
    
    # 3. 유지보수 일정 테스트
    print("\n3. 유지보수 일정 테스트")
    schedule = calculate_maintenance_schedule(
        last_maintenance=datetime.now() - timedelta(days=20),
        failure_probability=0.6,
        tool_wear=180
    )
    print(f"유지보수 일정: {schedule}")
    
    print("\n" + "=" * 60)
    print("테스트 완료!")
    print("=" * 60)

# =============================================================================
# 10. 메인 실행
# =============================================================================

if __name__ == "__main__":
    run_tests()