"""
4일차: AI Agent 설계 및 구현
=====================================

이 모듈은 OpenAI API를 활용하여 예측 모델 분석과 인사이트를 제공하는 
AI Agent를 구현합니다.

주요 기능:
1. 예측 결과 해석 및 설명
2. 데이터 패턴 분석
3. 모델 성능 비교 및 추천
4. 자연어 질의응답

Author: AI Agent Development Team
Date: 2024
"""

import os
import json
import pickle
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import openai
from dotenv import load_dotenv
import logging

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 환경 변수 로드
load_dotenv()

# =============================================================================
# 1. 데이터 클래스 정의
# =============================================================================

@dataclass
class ModelInfo:
    """모델 정보를 담는 데이터 클래스"""
    name: str
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    feature_importance: Dict[str, float]

@dataclass
class PredictionResult:
    """예측 결과를 담는 데이터 클래스"""
    model_name: str
    prediction: int
    probability: float
    confidence: str
    explanation: str

# =============================================================================
# 2. AI Agent 클래스
# =============================================================================

class ManufacturingAIAgent:
    """
    제조업 예측 모델을 위한 AI Agent
    
    이 클래스는 OpenAI API를 사용하여:
    - 모델 예측 결과를 해석
    - 데이터 패턴을 분석
    - 사용자 질문에 답변
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        AI Agent 초기화
        
        Args:
            api_key: OpenAI API 키 (없으면 환경변수에서 로드)
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API 키가 필요합니다. .env 파일을 확인하세요.")
        
        openai.api_key = self.api_key
        self.model_name = os.getenv("OPENAI_MODEL", "gpt-4-turbo-preview")
        self.max_tokens = int(os.getenv("MAX_TOKENS", "2000"))
        self.temperature = float(os.getenv("TEMPERATURE", "0.7"))
        
        # 모델 정보 로드 (사전 학습된 모델들)
        self.models_info = self._load_model_info()
        
        # 시스템 프롬프트 설정
        self.system_prompt = self._create_system_prompt()
        
        logger.info(f"AI Agent 초기화 완료 (모델: {self.model_name})")
    
    def _load_model_info(self) -> Dict[str, ModelInfo]:
        """
        사전 학습된 모델 정보 로드
        
        Returns:
            모델 정보 딕셔너리
        """
        # 실제 환경에서는 파일에서 로드하지만, 여기서는 예시 데이터 사용
        models_info = {
            "Random Forest": ModelInfo(
                name="Random Forest",
                accuracy=0.9825,
                precision=0.9024,
                recall=0.5441,
                f1_score=0.6789,
                feature_importance={
                    "Torque_Nm": 0.3705,
                    "Rotational_speed_rpm": 0.2359,
                    "Tool_wear_min": 0.1569,
                    "Air_temperature_K": 0.1059,
                    "Process_temperature_K": 0.1130,
                    "Type_encoded": 0.0179
                }
            ),
            "XGBoost": ModelInfo(
                name="XGBoost",
                accuracy=0.9885,
                precision=0.9245,
                recall=0.7206,
                f1_score=0.8099,
                feature_importance={
                    "Torque_Nm": 0.2958,
                    "Rotational_speed_rpm": 0.1780,
                    "Tool_wear_min": 0.1878,
                    "Air_temperature_K": 0.1580,
                    "Process_temperature_K": 0.0981,
                    "Type_encoded": 0.0822
                }
            ),
            "Deep Learning": ModelInfo(
                name="Deep Learning",
                accuracy=0.9910,
                precision=0.9500,
                recall=0.7800,
                f1_score=0.8565,
                feature_importance={}  # DL 모델은 전통적인 feature importance 없음
            )
        }
        return models_info
    
    def _create_system_prompt(self) -> str:
        """
        시스템 프롬프트 생성
        
        Returns:
            시스템 프롬프트 문자열
        """
        return """
        당신은 제조업 예측 유지보수 시스템의 AI 전문가입니다.
        
        주요 역할:
        1. 기계 고장 예측 결과를 분석하고 설명합니다
        2. 데이터 패턴과 이상 징후를 식별합니다
        3. 예방 유지보수 전략을 제안합니다
        4. 모델 성능을 비교하고 최적 모델을 추천합니다
        
        데이터셋 정보:
        - AI4I 2020 Predictive Maintenance Dataset
        - 10,000개 샘플, 제조 공정 데이터
        - 주요 특성: 온도, 회전속도, 토크, 도구 마모도
        - 고장 유형: TWF(도구마모), HDF(열방출), PWF(전력), OSF(과부하), RNF(랜덤)
        
        답변 원칙:
        - 기술적으로 정확하되 이해하기 쉽게 설명
        - 실무적인 인사이트 제공
        - 구체적인 수치와 예시 활용
        - 한국어로 답변
        """
    
    # =============================================================================
    # 3. 예측 해석 기능
    # =============================================================================
    
    def interpret_prediction(self, 
                            model_name: str,
                            prediction: int,
                            probability: float,
                            input_features: Dict[str, float]) -> PredictionResult:
        """
        모델 예측 결과를 해석
        
        Args:
            model_name: 모델 이름
            prediction: 예측값 (0: 정상, 1: 고장)
            probability: 예측 확률
            input_features: 입력 특성값
        
        Returns:
            해석된 예측 결과
        """
        # 신뢰도 계산
        confidence = self._calculate_confidence(probability)
        
        # 프롬프트 생성
        prompt = f"""
        다음 예측 결과를 분석하고 설명해주세요:
        
        모델: {model_name}
        예측: {'고장' if prediction == 1 else '정상'}
        확률: {probability:.2%}
        신뢰도: {confidence}
        
        입력 데이터:
        {json.dumps(input_features, indent=2)}
        
        다음 사항을 포함하여 설명해주세요:
        1. 예측 결과의 의미
        2. 주요 영향 요인 (feature importance 기반)
        3. 권장 조치사항
        """
        
        explanation = self._call_openai_api(prompt)
        
        return PredictionResult(
            model_name=model_name,
            prediction=prediction,
            probability=probability,
            confidence=confidence,
            explanation=explanation
        )
    
    def _calculate_confidence(self, probability: float) -> str:
        """
        예측 확률을 기반으로 신뢰도 계산
        
        Args:
            probability: 예측 확률
        
        Returns:
            신뢰도 레벨
        """
        if probability > 0.9:
            return "매우 높음"
        elif probability > 0.7:
            return "높음"
        elif probability > 0.5:
            return "보통"
        else:
            return "낮음"
    
    # =============================================================================
    # 4. 데이터 분석 기능
    # =============================================================================
    
    def analyze_data_pattern(self, df: pd.DataFrame) -> str:
        """
        데이터 패턴 분석
        
        Args:
            df: 분석할 데이터프레임
        
        Returns:
            패턴 분석 결과
        """
        # 기본 통계 계산
        stats = {
            "총 샘플 수": len(df),
            "고장률": df['Machine failure'].mean() * 100 if 'Machine failure' in df.columns else 0,
            "평균 온도": df['Air_temperature_K'].mean() if 'Air_temperature_K' in df.columns else 0,
            "평균 회전속도": df['Rotational_speed_rpm'].mean() if 'Rotational_speed_rpm' in df.columns else 0,
            "평균 토크": df['Torque_Nm'].mean() if 'Torque_Nm' in df.columns else 0,
        }
        
        prompt = f"""
        다음 제조 데이터의 패턴을 분석해주세요:
        
        {json.dumps(stats, indent=2, ensure_ascii=False)}
        
        다음 관점에서 분석해주세요:
        1. 전반적인 장비 상태
        2. 잠재적 위험 요소
        3. 개선 가능 영역
        4. 유지보수 우선순위
        """
        
        return self._call_openai_api(prompt)
    
    # =============================================================================
    # 5. 모델 비교 및 추천
    # =============================================================================
    
    def compare_models(self, test_results: Optional[Dict] = None) -> str:
        """
        모델 성능 비교 및 추천
        
        Args:
            test_results: 테스트 결과 (선택사항)
        
        Returns:
            비교 분석 결과
        """
        # 모델 정보를 텍스트로 변환
        models_text = ""
        for name, info in self.models_info.items():
            models_text += f"""
            {name}:
            - Accuracy: {info.accuracy:.4f}
            - Precision: {info.precision:.4f}
            - Recall: {info.recall:.4f}
            - F1-Score: {info.f1_score:.4f}
            """
        
        prompt = f"""
        다음 예측 모델들의 성능을 비교 분석해주세요:
        
        {models_text}
        
        다음 사항을 포함하여 분석해주세요:
        1. 각 모델의 장단점
        2. 사용 시나리오별 추천 모델
        3. 성능 개선 방안
        4. 종합적인 추천
        """
        
        return self._call_openai_api(prompt)
    
    # =============================================================================
    # 6. 대화형 분석 기능 (Enhanced)
    # =============================================================================
    
    def ask_question(self, question: str, context: Optional[Dict] = None) -> str:
        """
        사용자 질문에 답변
        
        Args:
            question: 사용자 질문
            context: 추가 컨텍스트 (선택사항)
        
        Returns:
            답변
        """
        context_text = ""
        if context:
            context_text = f"\n\n관련 정보:\n{json.dumps(context, indent=2, ensure_ascii=False)}"
        
        prompt = f"""
        사용자 질문: {question}
        {context_text}
        
        제조업 예측 유지보수 전문가로서 답변해주세요.
        """
        
        return self._call_openai_api(prompt)
    
    def analyze_natural_query(self, query: str, df: pd.DataFrame) -> Dict[str, Any]:
        """
        자연어 쿼리를 분석하여 데이터에서 정보 추출
        
        Args:
            query: 자연어 질의
            df: 데이터프레임
        
        Returns:
            분석 결과 딕셔너리
        """
        # 데이터 요약 정보
        data_summary = {
            "총 레코드": len(df),
            "컬럼": list(df.columns),
            "고장률": f"{df['Machine failure'].mean():.2%}" if 'Machine failure' in df.columns else "N/A",
            "기간": f"{df.index[0]} ~ {df.index[-1]}" if not df.empty else "N/A"
        }
        
        # 쿼리 타입 분석
        query_lower = query.lower()
        
        # 통계 관련 쿼리
        if any(word in query_lower for word in ['평균', '최대', '최소', '통계', 'average', 'max', 'min']):
            stats = self._analyze_statistics(df, query)
            return {"type": "statistics", "result": stats, "query": query}
        
        # 고장 관련 쿼리
        elif any(word in query_lower for word in ['고장', '실패', 'failure', 'breakdown']):
            failure_analysis = self._analyze_failures(df, query)
            return {"type": "failure_analysis", "result": failure_analysis, "query": query}
        
        # 트렌드 관련 쿼리
        elif any(word in query_lower for word in ['트렌드', '추세', '변화', 'trend', 'change']):
            trend_analysis = self._analyze_trends(df, query)
            return {"type": "trend_analysis", "result": trend_analysis, "query": query}
        
        # 일반 쿼리
        else:
            context = {"data_summary": data_summary}
            answer = self.ask_question(query, context)
            return {"type": "general", "result": answer, "query": query}
    
    def _analyze_statistics(self, df: pd.DataFrame, query: str) -> Dict:
        """통계 분석"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        stats = {}
        
        for col in numeric_cols:
            if col in ['Air_temperature_K', 'Process_temperature_K', 'Rotational_speed_rpm', 'Torque_Nm', 'Tool_wear_min']:
                stats[col] = {
                    "평균": df[col].mean(),
                    "최대": df[col].max(),
                    "최소": df[col].min(),
                    "표준편차": df[col].std()
                }
        
        prompt = f"""
        사용자가 다음 통계 정보에 대해 질문했습니다:
        질문: {query}
        
        데이터 통계:
        {json.dumps(stats, indent=2, ensure_ascii=False)}
        
        질문에 맞는 구체적인 답변을 제공하고, 인사이트를 추가해주세요.
        """
        
        explanation = self._call_openai_api(prompt)
        return {"statistics": stats, "explanation": explanation}
    
    def _analyze_failures(self, df: pd.DataFrame, query: str) -> Dict:
        """고장 분석"""
        if 'Machine failure' not in df.columns:
            return {"error": "고장 데이터가 없습니다"}
        
        failure_df = df[df['Machine failure'] == 1]
        failure_stats = {
            "총 고장 수": len(failure_df),
            "고장률": f"{(len(failure_df) / len(df)):.2%}",
            "고장 유형별": {}
        }
        
        # 고장 유형별 분석
        failure_types = ['TWF', 'HDF', 'PWF', 'OSF', 'RNF']
        for ftype in failure_types:
            if ftype in df.columns:
                failure_stats["고장 유형별"][ftype] = df[ftype].sum()
        
        # 고장 시 평균값
        if not failure_df.empty:
            failure_stats["고장 시 평균값"] = {
                "온도": failure_df['Air_temperature_K'].mean() if 'Air_temperature_K' in failure_df.columns else 0,
                "회전속도": failure_df['Rotational_speed_rpm'].mean() if 'Rotational_speed_rpm' in failure_df.columns else 0,
                "토크": failure_df['Torque_Nm'].mean() if 'Torque_Nm' in failure_df.columns else 0,
            }
        
        prompt = f"""
        사용자의 고장 관련 질문: {query}
        
        고장 분석 결과:
        {json.dumps(failure_stats, indent=2, ensure_ascii=False)}
        
        이 데이터를 바탕으로 질문에 답하고, 고장 예방 방안을 제시해주세요.
        """
        
        explanation = self._call_openai_api(prompt)
        return {"failure_stats": failure_stats, "explanation": explanation}
    
    def _analyze_trends(self, df: pd.DataFrame, query: str) -> Dict:
        """트렌드 분석"""
        # 시간 관련 컬럼이 있다면 사용, 없으면 인덱스 사용
        df['index'] = range(len(df))
        
        trends = {}
        numeric_cols = ['Air_temperature_K', 'Process_temperature_K', 'Rotational_speed_rpm', 'Torque_Nm', 'Tool_wear_min']
        
        for col in numeric_cols:
            if col in df.columns:
                # 간단한 트렌드 계산 (시작값 대비 종료값)
                start_val = df[col].iloc[:100].mean() if len(df) > 100 else df[col].iloc[0]
                end_val = df[col].iloc[-100:].mean() if len(df) > 100 else df[col].iloc[-1]
                change = ((end_val - start_val) / start_val * 100) if start_val != 0 else 0
                trends[col] = {
                    "시작값": start_val,
                    "종료값": end_val,
                    "변화율": f"{change:.2f}%"
                }
        
        prompt = f"""
        사용자의 트렌드 관련 질문: {query}
        
        트렌드 분석 결과:
        {json.dumps(trends, indent=2, ensure_ascii=False)}
        
        이 트렌드를 해석하고, 향후 예상되는 패턴을 설명해주세요.
        """
        
        explanation = self._call_openai_api(prompt)
        return {"trends": trends, "explanation": explanation}
    
    def what_if_scenario(self, 
                        scenario: str,
                        current_values: Dict[str, float],
                        models: Optional[Dict] = None) -> Dict[str, Any]:
        """
        What-if 시나리오 분석
        
        Args:
            scenario: 시나리오 설명
            current_values: 현재 값
            models: 예측 모델 (선택사항)
        
        Returns:
            시나리오 분석 결과
        """
        # 시나리오 파싱
        scenario_lower = scenario.lower()
        modified_values = current_values.copy()
        
        # 간단한 시나리오 파서
        if '온도' in scenario or 'temperature' in scenario_lower:
            # 온도 변경 시나리오
            if '증가' in scenario or 'increase' in scenario_lower or '상승' in scenario:
                change = 10  # 기본 10도 증가
                if any(str(i) in scenario for i in range(1, 100)):
                    import re
                    numbers = re.findall(r'\d+', scenario)
                    if numbers:
                        change = float(numbers[0])
                
                if 'Air_temperature_K' in modified_values:
                    modified_values['Air_temperature_K'] += change
                if 'Process_temperature_K' in modified_values:
                    modified_values['Process_temperature_K'] += change
        
        elif '회전' in scenario or 'rotation' in scenario_lower or 'speed' in scenario_lower:
            # 회전속도 변경 시나리오
            if '증가' in scenario or 'increase' in scenario_lower:
                if 'Rotational_speed_rpm' in modified_values:
                    modified_values['Rotational_speed_rpm'] *= 1.2  # 20% 증가
        
        elif '토크' in scenario or 'torque' in scenario_lower:
            # 토크 변경 시나리오
            if '증가' in scenario or 'increase' in scenario_lower:
                if 'Torque_Nm' in modified_values:
                    modified_values['Torque_Nm'] *= 1.3  # 30% 증가
        
        # 영향 분석
        impact_analysis = self._analyze_scenario_impact(current_values, modified_values)
        
        # 예측 변화 (모델이 제공된 경우)
        prediction_changes = {}
        if models:
            # 여기서는 간단한 규칙 기반 예측
            risk_score_before = self._calculate_risk_score(current_values)
            risk_score_after = self._calculate_risk_score(modified_values)
            prediction_changes = {
                "기존 위험도": f"{risk_score_before:.2%}",
                "변경 후 위험도": f"{risk_score_after:.2%}",
                "위험도 변화": f"{(risk_score_after - risk_score_before):.2%}"
            }
        
        # AI 해석
        prompt = f"""
        What-if 시나리오 분석:
        
        시나리오: {scenario}
        
        현재 값:
        {json.dumps(current_values, indent=2)}
        
        변경된 값:
        {json.dumps(modified_values, indent=2)}
        
        영향 분석:
        {json.dumps(impact_analysis, indent=2, ensure_ascii=False)}
        
        예측 변화:
        {json.dumps(prediction_changes, indent=2, ensure_ascii=False)}
        
        이 시나리오의 영향을 상세히 설명하고, 권장사항을 제시해주세요.
        """
        
        explanation = self._call_openai_api(prompt)
        
        return {
            "scenario": scenario,
            "original_values": current_values,
            "modified_values": modified_values,
            "impact_analysis": impact_analysis,
            "prediction_changes": prediction_changes,
            "explanation": explanation
        }
    
    def _analyze_scenario_impact(self, before: Dict, after: Dict) -> Dict:
        """시나리오 영향 분석"""
        impact = {}
        for key in before:
            if key in after and before[key] != after[key]:
                change = after[key] - before[key]
                pct_change = (change / before[key] * 100) if before[key] != 0 else 0
                impact[key] = {
                    "변경 전": before[key],
                    "변경 후": after[key],
                    "변화량": change,
                    "변화율": f"{pct_change:.2f}%"
                }
        return impact
    
    def _calculate_risk_score(self, values: Dict) -> float:
        """간단한 위험도 점수 계산"""
        score = 0.0
        
        # 온도 기반 위험도
        if 'Air_temperature_K' in values:
            if values['Air_temperature_K'] > 305:
                score += 0.2
        if 'Process_temperature_K' in values:
            if values['Process_temperature_K'] > 315:
                score += 0.2
        
        # 토크 기반 위험도
        if 'Torque_Nm' in values:
            if values['Torque_Nm'] > 50:
                score += 0.3
        
        # 도구 마모도 기반 위험도
        if 'Tool_wear_min' in values:
            if values['Tool_wear_min'] > 200:
                score += 0.3
        
        return min(score, 1.0)
    
    def compare_periods(self, 
                       df1: pd.DataFrame, 
                       df2: pd.DataFrame,
                       period1_name: str = "기간 1",
                       period2_name: str = "기간 2") -> Dict[str, Any]:
        """
        두 기간의 데이터를 비교 분석
        
        Args:
            df1: 첫 번째 기간 데이터
            df2: 두 번째 기간 데이터
            period1_name: 첫 번째 기간 이름
            period2_name: 두 번째 기간 이름
        
        Returns:
            비교 분석 결과
        """
        comparison = {
            period1_name: {},
            period2_name: {},
            "변화": {}
        }
        
        # 주요 메트릭 비교
        metrics = ['Air_temperature_K', 'Process_temperature_K', 'Rotational_speed_rpm', 'Torque_Nm', 'Tool_wear_min']
        
        for metric in metrics:
            if metric in df1.columns and metric in df2.columns:
                val1 = df1[metric].mean()
                val2 = df2[metric].mean()
                change = ((val2 - val1) / val1 * 100) if val1 != 0 else 0
                
                comparison[period1_name][metric] = round(val1, 2)
                comparison[period2_name][metric] = round(val2, 2)
                comparison["변화"][metric] = f"{change:.2f}%"
        
        # 고장률 비교
        if 'Machine failure' in df1.columns and 'Machine failure' in df2.columns:
            failure1 = df1['Machine failure'].mean()
            failure2 = df2['Machine failure'].mean()
            comparison[period1_name]["고장률"] = f"{failure1:.2%}"
            comparison[period2_name]["고장률"] = f"{failure2:.2%}"
            comparison["변화"]["고장률"] = f"{(failure2 - failure1):.2%}p"
        
        # AI 해석
        prompt = f"""
        두 기간의 데이터 비교 분석:
        
        {json.dumps(comparison, indent=2, ensure_ascii=False)}
        
        다음을 포함한 상세 분석을 제공해주세요:
        1. 주요 변화 사항
        2. 개선된 점과 악화된 점
        3. 변화의 원인 추정
        4. 향후 권장사항
        """
        
        explanation = self._call_openai_api(prompt)
        
        return {
            "comparison": comparison,
            "explanation": explanation
        }
    
    # =============================================================================
    # 7. OpenAI API 호출
    # =============================================================================
    
    def _call_openai_api(self, prompt: str) -> str:
        """
        OpenAI API 호출
        
        Args:
            prompt: 프롬프트
        
        Returns:
            API 응답
        """
        try:
            response = openai.ChatCompletion.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=self.max_tokens,
                temperature=self.temperature
            )
            
            return response.choices[0].message.content
        
        except Exception as e:
            logger.error(f"OpenAI API 호출 실패: {e}")
            return f"API 호출 중 오류가 발생했습니다: {str(e)}"
    
    # =============================================================================
    # 8. 예방 유지보수 전략
    # =============================================================================
    
    def generate_maintenance_strategy(self, 
                                     failure_history: List[Dict],
                                     current_status: Dict) -> str:
        """
        예방 유지보수 전략 생성
        
        Args:
            failure_history: 고장 이력
            current_status: 현재 상태
        
        Returns:
            유지보수 전략
        """
        prompt = f"""
        다음 정보를 기반으로 예방 유지보수 전략을 수립해주세요:
        
        고장 이력:
        {json.dumps(failure_history, indent=2, ensure_ascii=False)}
        
        현재 상태:
        {json.dumps(current_status, indent=2, ensure_ascii=False)}
        
        다음 사항을 포함한 전략을 제시해주세요:
        1. 즉시 조치사항
        2. 단기 계획 (1주일)
        3. 중기 계획 (1개월)
        4. 장기 계획 (3개월)
        5. 비용-효과 분석
        """
        
        return self._call_openai_api(prompt)

# =============================================================================
# 9. 유틸리티 함수
# =============================================================================

def load_model(model_path: str):
    """저장된 모델 로드"""
    with open(model_path, 'rb') as f:
        return pickle.load(f)

def prepare_input_features(data: Dict[str, float]) -> pd.DataFrame:
    """입력 데이터를 모델에 맞게 전처리"""
    # 필요한 특성 순서
    feature_order = [
        'Air_temperature_K',
        'Process_temperature_K', 
        'Rotational_speed_rpm',
        'Torque_Nm',
        'Tool_wear_min',
        'Type_encoded'
    ]
    
    # DataFrame 생성
    df = pd.DataFrame([data])
    
    # 특성 순서 맞추기
    return df[feature_order]

# =============================================================================
# 10. 실습 예제
# =============================================================================

if __name__ == "__main__":
    """
    실습 예제: AI Agent 사용 방법
    """
    
    print("=" * 60)
    print("AI Agent 실습 예제")
    print("=" * 60)
    
    try:
        # 1. AI Agent 초기화
        print("\n1. AI Agent 초기화...")
        agent = ManufacturingAIAgent()
        print("✅ Agent 초기화 완료")
        
        # 2. 예측 해석 예제
        print("\n2. 예측 해석 예제")
        print("-" * 40)
        
        # 테스트 데이터
        test_features = {
            "Air_temperature_K": 301.5,
            "Process_temperature_K": 311.2,
            "Rotational_speed_rpm": 1450,
            "Torque_Nm": 45.3,
            "Tool_wear_min": 180,
            "Type": "H"
        }
        
        # 예측 결과 해석
        result = agent.interpret_prediction(
            model_name="XGBoost",
            prediction=1,
            probability=0.85,
            input_features=test_features
        )
        
        print(f"모델: {result.model_name}")
        print(f"예측: {'고장' if result.prediction == 1 else '정상'}")
        print(f"확률: {result.probability:.2%}")
        print(f"신뢰도: {result.confidence}")
        print(f"\n설명:\n{result.explanation[:500]}...")
        
        # 3. 모델 비교
        print("\n3. 모델 성능 비교")
        print("-" * 40)
        comparison = agent.compare_models()
        print(comparison[:500] + "...")
        
        # 4. 질의응답
        print("\n4. 질의응답 예제")
        print("-" * 40)
        
        questions = [
            "토크가 높을 때 어떤 조치를 취해야 하나요?",
            "도구 마모도가 200분을 넘으면 위험한가요?",
            "어떤 모델을 실제 운영 환경에서 사용하면 좋을까요?"
        ]
        
        for q in questions[:1]:  # 예제로 첫 번째 질문만
            print(f"\nQ: {q}")
            answer = agent.ask_question(q)
            print(f"A: {answer[:300]}...")
        
        print("\n" + "=" * 60)
        print("실습 완료! 🎉")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")
        print("\n해결 방법:")
        print("1. .env 파일을 생성하고 OPENAI_API_KEY를 설정하세요")
        print("2. requirements.txt의 패키지를 모두 설치하세요")
        print("3. API 키가 유효한지 확인하세요")

# =============================================================================
# 실습 과제
# =============================================================================
"""
💡 실습 과제:

1. **커스텀 프롬프트 작성**
   - 특정 고장 유형에 특화된 프롬프트 작성
   - 다국어 지원 추가

2. **컨텍스트 관리 개선**
   - 대화 히스토리 저장 및 활용
   - 세션 관리 구현

3. **성능 최적화**
   - API 호출 캐싱
   - 배치 처리 구현

4. **확장 기능**
   - 이미지 분석 (GPT-4 Vision API)
   - 음성 인터페이스 추가
"""