"""
MedGemma 의료 특화 LLM 검색 담당 클래스 - Hugging Face Inference Endpoints 사용
"""

from typing import List, Dict, Any, Optional
from langchain_core.documents import Document
from datetime import datetime
import requests
import json
import time
import os
import sys
from pathlib import Path
from dotenv import load_dotenv
import logging
logger = logging.getLogger(__name__)

from prompts import system_prompts

ROOT_DIR = Path(__file__).resolve().parent.parent
dotenv_path = os.path.join(ROOT_DIR, '.env')

class MedGemmaSearcher:
    """MedGemma 의료 특화 LLM 검색 담당 클래스 (Hugging Face 기반)"""
    
    def __init__(self, 
                model_name: str = "google/medgemma_27b_text_it", 
                hf_api_key: str = None, 
                inference_endpoint_url: str = None):
        """
        MedGemma 검색기 초기화
        
        Args:
            model_name: 사용할 모델명 (의료 파인튜닝 버전 권장)
            hf_api_key: Hugging Face API 키
            inference_endpoint_url: Inference Endpoint URL
        """
        # API 키 설정
        self.hf_api_key = hf_api_key or os.getenv("HUGGINGFACE_API_KEY")
        if not self.hf_api_key:
            logger.warning("Hugging Face API 키가 설정되지 않았습니다.")
        
        # 엔드포인트 URL 설정
        self.inference_endpoint = inference_endpoint_url or os.getenv("HF_INFERENCE_ENDPOINT")
        if not self.inference_endpoint:
            logger.warning("Inference Endpoint URL이 설정되지 않았습니다.")
        
        self.model_name = model_name

        # 검색 통계
        self.search_stats = {
            "queries_processed": 0,
            "successful_generations": 0,
            "failed_generations": 0,
            "average_response_length": 0,
            "total_tokens_generated": 0,
            "api_errors": 0,
            "average_latency": 0
        }
        
        print("🧠 MedGemma 초기화 중...")
        self.model_loaded = self._check_endpoint_connection()
        if self.model_loaded:
            print("✅ MedGemma 초기화 완료")
        else:
            print("⚠️ MedGemma 초기화 실패")
    
    def _check_endpoint_connection(self) -> bool:
        """Inference Endpoint 연결 확인"""
        if not self.hf_api_key or not self.inference_endpoint:
            print("❌ API 키 또는 엔드포인트 URL이 설정되지 않았습니다.")
            return False
        
        try:
            # 간단한 테스트 요청
            headers = {
                "Authorization": f"Bearer {self.hf_api_key}",
                "Content-Type": "application/json"
            }
            
            test_payload = {
                "inputs": "안녕하세요",
                "parameters": {
                    "max_new_tokens": 10,
                    "temperature": 0.7
                }
            }
            
            print("🔄 Inference Endpoint 연결 테스트 중...")
            response = requests.post(
                self.inference_endpoint, 
                headers=headers, 
                json=test_payload,
                timeout=10
            )
            
            if response.status_code == 200:
                print("✅ Inference Endpoint 연결 성공")
                return True
            else:
                print(f"❌ Inference Endpoint 연결 실패: {response.status_code}")
                print(f"응답: {response.text}")
                return False
                
        except Exception as e:
            print(f"❌ Inference Endpoint 연결 테스트 실패: {str(e)}")
            return False
    
    def search_medgemma(self, query: str, max_results: int = 1, max_length: int = 1536) -> List[Document]:
        """MedGemma를 사용한 의료 지식 검색"""
        print(f"==== [MEDGEMMA SEARCH: {query}] ====")
        
        start_time = time.time()
        self.search_stats["queries_processed"] += 1
        
        if not self.model_loaded:
            print("  ❌ MedGemma 모델이 로드되지 않음")
            return self._create_fallback_documents(query, "model_not_loaded")
        
        try:
            prompt_template = system_prompts.get("MEDGEMMA")
                        # 의료 특화 프롬프트 구성

            if prompt_template is None:
                prompt_template = f"""<s>[INST] 다음은 의료진을 위한 질문입니다. 상세하고 정확한 답변을 한국어로 제공해주세요.

질문: {query}

답변을 할 때 다음 사항을 고려하세요:
- 정확한 의학 용어를 사용하세요
- 필요시 부작용이나 금기사항을 언급하세요
- 응급 상황일 경우 중요도를 강조하세요
- 단계별 지침이 필요한 경우 명확히 순서를 제시하세요 [/INST]"""
                
            
            # 질문 유형 감지 (이 기능을 유지하려면)
            question_type = self._detect_medical_question_type(query)
            
            # 템플릿에 직접 변수 적용
            prompt = prompt_template.format(query=query)

            # MedGemma 추론 실행 (바로 API 호출)
            response = self._generate_medical_response(prompt, max_length)
                
            if response and len(response.strip()) > 10:  # 최소 길이 확인
                # Document 객체로 변환
                document = self._convert_to_document(query, response)
                
                # 통계 업데이트
                end_time = time.time()
                latency = end_time - start_time
                
                self.search_stats["successful_generations"] += 1
                self.search_stats["total_tokens_generated"] += len(response.split())
                self._update_latency_stats(latency)
                
                print(f"  ✅ MedGemma 응답 생성 완료 ({len(response)}자)")
                print(f"  ⏱️ 응답 시간: {latency:.2f}초")
                
                return [document]
            else:
                print(f"  ❌ MedGemma 응답이 너무 짧음: '{response}'")
                self.search_stats["failed_generations"] += 1
                return self._create_fallback_documents(query, "short_response")
                
        except Exception as e:
            logger.error(f"MedGemma 검색 실패: {str(e)}")
            print(f"  ❌ MedGemma 오류: {str(e)}")
            self.search_stats["failed_generations"] += 1
            self.search_stats["api_errors"] += 1
            return self._create_fallback_documents(query, str(e))
    
    def _detect_medical_question_type(self, query: str) -> str:
        """의료 질문 유형 감지"""
        query_lower = query.lower()
        
        # 키워드 기반 분류
        if any(word in query_lower for word in ["응급", "급성", "심정지", "쇼크", "출혈"]):
            return "응급상황"
        elif any(word in query_lower for word in ["진단", "검사", "증상", "원인"]):
            return "진단정보"
        elif any(word in query_lower for word in ["치료", "처치", "관리", "요법"]):
            return "치료정보"
        elif any(word in query_lower for word in ["약물", "약", "처방", "부작용"]):
            return "약물정보"
        elif any(word in query_lower for word in ["수술", "시술", "절차", "프로토콜"]):
            return "시술정보"
        else:
            return "일반의학정보"
    
    def _generate_medical_response(self, prompt: str, max_length: int) -> Optional[str]:
        """MedGemma를 사용한 의료 응답 생성"""
        try:
            headers = {
                "Authorization": f"Bearer {self.hf_api_key}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "inputs": prompt,
                "parameters": {
                    "max_new_tokens": max_length,
                    "temperature": 0.7,
                    "top_p": 0.9,
                    "do_sample": True,
                    "return_full_text": False
                }
            }
            
            print(f"    🔄 MedGemma 추론 시작... (max_tokens: {max_length})")
            print(f"    📋 프롬프트: '{prompt[:200]}...'")
            
            response = requests.post(
                self.inference_endpoint, 
                headers=headers, 
                json=payload,
                timeout=120
            )
            
            if response.status_code == 200:
                try:
                    result = response.json()
                    
                    # 응답 구조에 따라 결과 추출
                    if isinstance(result, list) and len(result) > 0:
                        generated_text = result[0].get("generated_text", "")
                    elif isinstance(result, dict):
                        generated_text = result.get("generated_text", "")
                    else:
                        generated_text = str(result)

                    # 원본 응답 출력 
                    print("\n==== MEDGEMMA 원본 응답 시작 ====")
                    print(f"응답 형식: {type(result)}")
                    print(f"응답 내용: {json.dumps(result, indent=2, ensure_ascii=False)}")
                    print("==== MEDGEMMA 원본 응답 끝 ====\n")
                                
                    print(f"    📝 원본 응답: '{generated_text[:100]}...'")
                    print(f"    📝 원본 응답 길이: {len(generated_text)}자")
                    
                    # 응답 후처리
                    cleaned_response = self._clean_medical_response(generated_text)
                    print(f"    ✨ 정리된 응답 길이: {len(cleaned_response)}자")
                    
                    return cleaned_response
                    
                except Exception as e:
                    logger.error(f"응답 파싱 실패: {str(e)}, 응답: {response.text}")
                    return None
            else:
                logger.error(f"API 요청 실패: {response.status_code}, 응답: {response.text}")
                print(f"    ❌ API 요청 실패: {response.status_code}")
                return None
                
        except requests.exceptions.Timeout:
            logger.error("API 요청 타임아웃")
            print(f"    ❌ API 요청 타임아웃")
            return None
        except Exception as e:
            logger.error(f"API 호출 실패: {str(e)}")
            print(f"    ❌ API 호출 실패: {str(e)}")
            return None
    
    def _clean_medical_response(self, response: str) -> str:
        """생성된 의료 응답 정리"""
        if not response:
            return ""
        
        # 불필요한 토큰 제거
        clean_response = response.strip()
        
        # 반복 패턴 제거
        lines = clean_response.split('\n')
        unique_lines = []
        seen_lines = set()
        
        for line in lines:
            line_clean = line.strip()
            if line_clean and line_clean not in seen_lines:
                unique_lines.append(line)
                seen_lines.add(line_clean)
        
        clean_response = '\n'.join(unique_lines)
        
        # 의료 정보 검증 마크 추가
        if len(clean_response) > 50:
            clean_response += "\n\n⚠️ 이 정보는 AI가 생성한 것으로, 실제 진료 시에는 반드시 의료진과 상담하시기 바랍니다."
        
        return clean_response
    
    def _convert_to_document(self, query: str, response: str) -> Document:
        """MedGemma 응답을 Document 객체로 변환"""
        
        # 응답 품질 평가
        quality_score = self._assess_response_quality(response)
        
        # 의료 카테고리 추정
        estimated_category = self._estimate_medical_category(query, response)
        
        # Document 생성
        content = f"""MedGemma 의료 지식 응답:

질문: {query}

답변:
{response}

생성 정보:
- 모델: {self.model_name}
- 생성 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- 품질 점수: {quality_score}/10
"""

        metadata = {
            "source": f"medgemma_{self.model_name}",
            "source_type": "medgemma",
            "model_name": self.model_name,
            "query": query,
            "generated_at": datetime.now().isoformat(),
            "api_type": "huggingface_inference",
            "quality_score": quality_score,
            "estimated_category": estimated_category,
            "response_length": len(response),
            "reliability": "high",  # MedGemma는 의료 특화 모델
            "confidence": "high" if quality_score >= 7 else "medium"
        }
        
        return Document(page_content=content, metadata=metadata)
    
    def _assess_response_quality(self, response: str) -> float:
        """응답 품질 평가 (1-10점)"""
        score = 5.0  # 기본 점수
        
        # 길이 평가
        if 100 <= len(response) <= 1000:
            score += 1.0
        elif len(response) > 1000:
            score += 0.5
        
        # 의료 용어 포함 여부
        medical_terms = ["치료", "진단", "증상", "약물", "처방", "환자", "의료진", "병원"]
        term_count = sum(1 for term in medical_terms if term in response)
        score += min(2.0, term_count * 0.3)
        
        # 구조화된 정보 여부 (번호, 단계 등)
        if any(pattern in response for pattern in ["1.", "2.", "첫째", "둘째", "단계"]):
            score += 1.0
        
        # 안전 정보 포함 여부
        if any(word in response for word in ["주의", "경고", "부작용", "금기"]):
            score += 1.0
        
        return min(10.0, score)
    
    def _estimate_medical_category(self, query: str, response: str) -> str:
        """의료 카테고리 추정"""
        combined_text = f"{query} {response}".lower()
        
        category_keywords = {
            "응급처치": ["응급", "급성", "심정지", "응급처치"],
            "내과": ["당뇨", "고혈압", "내과", "만성질환"],
            "외과": ["수술", "외과", "절개", "봉합"],
            "약물정보": ["약물", "처방", "부작용", "용법"],
            "진단검사": ["진단", "검사", "영상", "혈액"],
            "감염관리": ["감염", "항생제", "바이러스", "세균"]
        }
        
        max_matches = 0
        best_category = "일반의학"
        
        for category, keywords in category_keywords.items():
            matches = sum(1 for keyword in keywords if keyword in combined_text)
            if matches > max_matches:
                max_matches = matches
                best_category = category
        
        return best_category
    
    def _create_fallback_documents(self, query: str, reason: str = "unknown") -> List[Document]:
        """MedGemma 실패 시 폴백 문서 생성"""
        fallback_content = f"""
MedGemma 모델을 사용할 수 없어 기본 의료 정보를 제공합니다.

질문: {query}

일반적인 의료 가이드라인:
1. 정확한 진단을 위해서는 의료진과 직접 상담하시기 바랍니다
2. 응급상황 시에는 즉시 119에 신고하세요
3. 약물 복용 전에는 반드시 전문의와 상의하세요
4. 증상이 지속되거나 악화되면 병원 방문을 권장합니다

⚠️ 이는 MedGemma 모델 오류로 인한 기본 안내사항입니다.
정확한 의료 정보는 의료 전문가와 상담하세요.

오류 이유: {reason}
"""
        
        return [Document(
            page_content=fallback_content,
            metadata={
                "source": "medgemma_fallback",
                "source_type": "medgemma",
                "query": query,
                "fallback_reason": reason,
                "reliability": "low",
                "confidence": "low",
                "generated_at": datetime.now().isoformat()
            }
        )]
    
    def _update_latency_stats(self, latency: float):
        """응답 시간 통계 업데이트"""
        current_avg = self.search_stats["average_latency"]
        request_count = self.search_stats["successful_generations"]
        
        if request_count <= 1:
            self.search_stats["average_latency"] = latency
        else:
            new_avg = ((current_avg * (request_count - 1)) + latency) / request_count
            self.search_stats["average_latency"] = new_avg
    
    def get_stats(self) -> Dict[str, Any]:
        """MedGemma 검색기 통계"""
        success_rate = 0
        if self.search_stats["queries_processed"] > 0:
            success_rate = self.search_stats["successful_generations"] / self.search_stats["queries_processed"]
        
        avg_length = 0
        if self.search_stats["successful_generations"] > 0:
            avg_length = self.search_stats["total_tokens_generated"] / self.search_stats["successful_generations"]
        
        return {
            "searcher_type": "MedGemmaSearcher",
            "model_info": {
                "model_name": self.model_name,
                "api_type": "huggingface_inference",
                "model_loaded": self.model_loaded,
                "endpoint_configured": bool(self.inference_endpoint and self.hf_api_key)
            },
            "performance": {
                "queries_processed": self.search_stats["queries_processed"],
                "successful_generations": self.search_stats["successful_generations"],
                "failed_generations": self.search_stats["failed_generations"],
                "api_errors": self.search_stats["api_errors"],
                "success_rate": round(success_rate * 100, 2),
                "average_response_length": round(avg_length, 1),
                "average_latency": round(self.search_stats["average_latency"], 2)
            },
            "resource_usage": {
                "total_tokens_generated": self.search_stats["total_tokens_generated"],
                "estimated_cost_usd": round(self.search_stats["total_tokens_generated"] * 0.0001, 4)  # 추정 비용
            }
        }
    
    def cleanup(self):
        """리소스 정리"""
        print("🗑️ MedGemma 리소스 정리 완료")
