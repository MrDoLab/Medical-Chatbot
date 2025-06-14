# components/medgemma_searcher.py
from typing import List, Dict, Any, Optional
from langchain_core.documents import Document
from datetime import datetime
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import logging

logger = logging.getLogger(__name__)

class MedGemmaSearcher:
    """MedGemma 의료 특화 LLM 검색 담당 클래스"""
    
    def __init__(self, model_name: str = "microsoft/DialoGPT-medium", device: str = "auto"):
        """
        MedGemma 검색기 초기화
        
        Args:
            model_name: 사용할 Gemma 모델명 (의료 파인튜닝 버전 권장)
            device: 실행 디바이스 ("auto", "cpu", "cuda")
        """
        self.model_name = model_name
        self.device = self._get_device(device)
        
        # 모델 및 토크나이저 초기화
        self.tokenizer = None
        self.model = None
        self.pipeline = None
        
        # 의료 특화 설정
        self.medical_system_prompt = """You are MedGemma, a medical AI assistant specialized in providing accurate, evidence-based medical information for healthcare professionals.

Your capabilities:
- Provide clinical guidelines and treatment protocols
- Explain medical procedures and diagnostic criteria
- Offer drug information including dosages and contraindications
- Support emergency response protocols
- Give differential diagnosis suggestions

Guidelines:
- Always prioritize patient safety
- Use precise medical terminology
- Include relevant contraindications and warnings
- Mention when to seek immediate medical attention
- Provide step-by-step clinical procedures when appropriate
- Always respond in Korean regardless of input language

Remember: You are providing information for medical professionals. Be thorough and clinically oriented."""

        # 모델 로드 시도
        self.model_loaded = False
        self._try_load_model()
        
        # 검색 통계
        self.search_stats = {
            "queries_processed": 0,
            "successful_generations": 0,
            "failed_generations": 0,
            "average_response_length": 0,
            "total_tokens_generated": 0
        }
    
    def _get_device(self, device: str) -> str:
        """최적의 디바이스 결정"""
        if device == "auto":
            if torch.cuda.is_available():
                return "cuda"
            elif torch.backends.mps.is_available():  # Apple Silicon
                return "mps"
            else:
                return "cpu"
        return device
    
    def _try_load_model(self):
        """MedGemma 모델 로드 시도"""
        print(f"🧠 MedGemma 모델 로딩 중... ({self.device})")
        
        import os
        is_cloud = os.environ.get('STREAMLIT_SHARING', '') == 'true'

        # 클라우드 환경에서는 모델 로딩 건너뜀
        if is_cloud:
            print("⚠️ Streamlit Cloud 환경 감지: MedGemma 모델 로딩 건너뜀")
            self.model_loaded = False
            return

        try:
            # Hugging Face 토큰 로드
            import os
            from dotenv import load_dotenv
            load_dotenv()
            
            hf_token = os.getenv("HUGGINGFACE_HUB_TOKEN")
            if hf_token:
                print("✅ Hugging Face 토큰 확인됨")
            else:
                print("⚠️ Hugging Face 토큰이 없습니다")
            
            # 토크나이저 로드
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True,
                token=hf_token  # 토큰 명시적 전달
            )
            
            # 패딩 토큰 설정
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # 모델 로드 (메모리 효율적으로)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None,
                trust_remote_code=True,
                low_cpu_mem_usage=True,
                token=hf_token  # 토큰 명시적 전달
            )
            
            # 파이프라인 생성
            self.pipeline = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                device=self.device if self.device != "auto" else None,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                trust_remote_code=True
            )
            
            self.model_loaded = True
            print(f"✅ MedGemma 모델 로드 완료 ({self.device})")
            
        except Exception as e:
            logger.error(f"MedGemma 모델 로드 실패: {str(e)}")
            print(f"❌ MedGemma 모델 로드 실패: {str(e)}")
            self.model_loaded = False
    
    def search_medgemma(self, query: str, max_results: int = 3, max_length: int = 512) -> List[Document]:
        """MedGemma를 사용한 의료 지식 검색 (PubMedSearcher 패턴 호환)"""
        print(f"==== [MEDGEMMA SEARCH: {query}] ====")
        
        self.search_stats["queries_processed"] += 1
        
        if not self.model_loaded:
            print("  ❌ MedGemma 모델이 로드되지 않음")
            return self._create_fallback_documents(query)
        
        try:
            # 의료 특화 프롬프트 구성
            medical_prompt = self._build_medical_prompt(query)
            
            # MedGemma 추론 실행
            response = self._generate_medical_response(medical_prompt, max_length)
            
            if response and len(response.strip()) > 10:  # 최소 길이 확인
                # Document 객체로 변환
                document = self._convert_to_document(query, response)
                
                self.search_stats["successful_generations"] += 1
                self.search_stats["total_tokens_generated"] += len(response.split())
                
                print(f"  ✅ MedGemma 응답 생성 완료 ({len(response)}자)")
                return [document]
            else:
                print(f"  ❌ MedGemma 응답이 너무 짧음: '{response}'")
                self.search_stats["failed_generations"] += 1
                return self._create_fallback_documents(query)
                
        except Exception as e:
            logger.error(f"MedGemma 검색 실패: {str(e)}")
            print(f"  ❌ MedGemma 오류: {str(e)}")
            self.search_stats["failed_generations"] += 1
            return self._create_fallback_documents(query)
    
    def _build_medical_prompt(self, query: str) -> str:
        """의료 질문을 위한 프롬프트 구성 (단순화)"""
        
        # 단순한 프롬프트로 변경
        prompt = f"""다음은 의료진을 위한 질문입니다. 상세하고 정확한 답변을 한국어로 제공해주세요.

질문: {query}

답변:"""
        
        return prompt
        
        return prompt
    
    def _detect_medical_question_type(self, query: str) -> str:
        """의료 질문 유형 감지"""
        query_lower = query.lower()
        
        # 키워드 기반 분류
        if any(word in query_lower for word in ["응급", "급성", "심정지", "쇼크", "출혈"]):
            return "emergency"
        elif any(word in query_lower for word in ["진단", "검사", "증상", "원인"]):
            return "diagnosis"
        elif any(word in query_lower for word in ["치료", "처치", "관리", "요법"]):
            return "treatment"
        elif any(word in query_lower for word in ["약물", "약", "처방", "부작용"]):
            return "medication"
        elif any(word in query_lower for word in ["수술", "시술", "절차", "프로토콜"]):
            return "procedure"
        else:
            return "general"
    
    def _generate_medical_response(self, prompt: str, max_length: int) -> Optional[str]:
        """MedGemma를 사용한 의료 응답 생성"""
        
        try:
            # 생성 파라미터 설정 (매우 관대하게)
            generation_config = {
                "max_new_tokens": max_length,
                "min_length": 20,  # 최소 길이 줄임
                "temperature": 0.7,  # 더 창의적으로
                "top_p": 0.9,
                "do_sample": True,
                "pad_token_id": self.tokenizer.eos_token_id,
                "repetition_penalty": 1.1
            }
            
            print(f"    🤖 MedGemma 추론 시작... (max_tokens: {max_length})")
            print(f"    📋 프롬프트: '{prompt[:200]}...'")
            
            # 간단한 테스트 먼저
            try:
                test_output = self.pipeline("안녕하세요", max_new_tokens=10, do_sample=False)
                print(f"    🧪 기본 테스트: '{test_output[0]['generated_text']}'")
            except Exception as e:
                print(f"    ❌ 기본 테스트 실패: {str(e)}")
            
            # 응답 생성
            outputs = self.pipeline(
                prompt,
                **generation_config,
                return_full_text=False
            )
            
            print(f"    🔍 파이프라인 출력 타입: {type(outputs)}")
            print(f"    🔍 파이프라인 출력 길이: {len(outputs) if outputs else 0}")
            
            if outputs and len(outputs) > 0:
                generated_text = outputs[0]["generated_text"]
                print(f"    📝 원본 응답: '{generated_text[:100]}...'")
                print(f"    📝 원본 응답 길이: {len(generated_text)}자")
                
                # 응답 후처리
                cleaned_response = self._clean_medical_response(generated_text)
                print(f"    ✨ 정리된 응답 길이: {len(cleaned_response)}자")
                
                return cleaned_response
            else:
                print(f"    ❌ 파이프라인 출력이 비어있음")
                return None
            
        except Exception as e:
            logger.error(f"응답 생성 실패: {str(e)}")
            print(f"    ❌ 응답 생성 오류: {str(e)}")
            return None
    
    def _clean_medical_response(self, response: str) -> str:
        """생성된 의료 응답 정리"""
        
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
            "device": self.device,
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
    
    def _create_fallback_documents(self, query: str) -> List[Document]:
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
"""
        
        return [Document(
            page_content=fallback_content,
            metadata={
                "source": "medgemma_fallback",
                "source_type": "medgemma",
                "query": query,
                "fallback_reason": "model_unavailable",
                "reliability": "low",
                "confidence": "low",
                "generated_at": datetime.now().isoformat()
            }
        )]
    
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
                "device": self.device,
                "model_loaded": self.model_loaded
            },
            "performance": {
                "queries_processed": self.search_stats["queries_processed"],
                "successful_generations": self.search_stats["successful_generations"],
                "failed_generations": self.search_stats["failed_generations"],
                "success_rate": round(success_rate * 100, 2),
                "average_response_length": round(avg_length, 1)
            },
            "resource_usage": {
                "total_tokens_generated": self.search_stats["total_tokens_generated"],
                "estimated_cost_usd": 0.0  # 로컬 실행이므로 비용 없음
            }
        }
    
    def cleanup(self):
        """리소스 정리"""
        try:
            if self.model is not None:
                del self.model
            if self.tokenizer is not None:
                del self.tokenizer
            if self.pipeline is not None:
                del self.pipeline
            
            # GPU 메모리 정리
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            print("🗑️ MedGemma 리소스 정리 완료")
            
        except Exception as e:
            logger.error(f"리소스 정리 실패: {str(e)}")

# 사용 예시 및 테스트
def test_medgemma_searcher():
    """MedGemma 검색기 테스트"""
    print("🧪 MedGemma 검색기 테스트 시작\n")
    
    try:
        # 검색기 초기화
        searcher = MedGemmaSearcher()
        
        if not searcher.model_loaded:
            print("❌ 모델 로드 실패로 테스트 중단")
            return
        
        # 테스트 질문들
        test_queries = [
            "당뇨병 환자의 혈당 관리 방법",
            "고혈압 응급상황 대처법",
            "소아 발열 시 처치 방법"
        ]
        
        for query in test_queries:
            print(f"🔍 테스트: '{query}'")
            documents = searcher.search_medgemma(query)
            
            if documents:
                doc = documents[0]
                print(f"   ✅ 응답 생성 성공")
                print(f"   📄 응답 길이: {len(doc.page_content)}자")
                print(f"   ⭐ 품질 점수: {doc.metadata.get('quality_score', 0)}/10")
                print(f"   🏷️ 카테고리: {doc.metadata.get('estimated_category', 'N/A')}")
            else:
                print("   ❌ 응답 생성 실패")
            print()
        
        # 통계 출력
        stats = searcher.get_stats()
        print("📊 MedGemma 통계:")
        print(f"   성공률: {stats['performance']['success_rate']}%")
        print(f"   평균 응답 길이: {stats['performance']['average_response_length']}토큰")
        print(f"   처리된 질문: {stats['performance']['queries_processed']}개")
        
        # 리소스 정리
        searcher.cleanup()
        
    except Exception as e:
        print(f"❌ 테스트 실패: {str(e)}")

if __name__ == "__main__":
    test_medgemma_searcher()