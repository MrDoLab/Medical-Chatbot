# config.py (업데이트된 버전)
import os
from dotenv import load_dotenv
from typing import Dict, Any

# .env 파일 자동 로딩 (모든 파일에서 사용 가능)
load_dotenv()

class Config:
    """RAG 시스템 설정 클래스"""
    
    # LLM 설정
    MODEL_NAME = "gpt-4o"
    TEMPERATURE = 0
    
    # OpenAI 임베딩 설정
    EMBEDDING_CONFIG = {
        "model": "text-embedding-3-large",
        "dimensions": None,  # 기본 3072차원 사용 (최고 성능)
        "cache_enabled": True,
        "cache_ttl_days": 7,  # 캐시 유효 기간
        "batch_size": 100,  # 배치 처리 크기
        "max_retries": 3  # API 실패 시 재시도 횟수
    }
    
    # 검색 설정
    SEARCH_CONFIG = {
    "top_k": 5,  # 기본 검색 결과 개수
    "similarity_threshold": 0.3,  # 유사도 임계값
    "medical_keywords_boost": True,  # 의료 키워드 가중치 부여
    "category_filtering": True,  # 카테고리 기반 필터링
    "emergency_priority": True,  # 응급 문서 우선순위
    "enable_web_search": True,   # 웹 검색 활성화
    "reliable_domains": [        # 신뢰할 수 있는 의료 도메인
        "pubmed.ncbi.nlm.nih.gov", "mayoclinic.org", "who.int", 
        "cdc.gov", "nih.gov", "medlineplus.gov", "uptodate.com"
    ]
}
    
    # 유사도 임계값 (직접 접근용)
    SIMILARITY_THRESHOLD = 0.3
    
    # 벡터 스토어 설정 (레거시 - 사용 안함)
    VECTORSTORE_CONFIG = {
        "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
        "persist_directory": "./chroma_db", 
        "search_kwargs": {"k": 3}
    }
    
    # 재작성 설정
    MAX_REWRITE_COUNT = 2  # 환각 체크 최대 재시도 횟수
    
    # 재귀 한도 설정  
    RECURSION_LIMIT = 50  # 간소화된 워크플로우로 줄임
    
    # 의료 도메인 특화 설정
    MEDICAL_CONFIG = {
        "categories": [
            "응급처치", "내과", "외과", "소아과", "산부인과", 
            "정신과", "영상의학", "병리학", "약학", "간호학",
            "만성질환관리", "약물정보", "진단검사", "감염관리"
        ],
        "severity_levels": ["low", "medium", "high", "critical"],
        "departments": [
            "응급의학과", "내과", "외과", "소아과", "산부인과",
            "정형외과", "신경과", "정신건강의학과", "영상의학과"
        ]
    }

    # MedGemma 설정
    MEDGEMMA_CONFIG = {
    "model_name": "google/medgemma-27b-it",
    "device": "auto",
    "max_response_length": 1024,  
    "enable_medgemma": True,
    "fallback_on_error": True,
    "quantization": "4bit",  
    "model_kwargs": {
        "low_cpu_mem_usage": True,
        "load_in_4bit": True,
        "device_map": "auto"
    }
}
    
    # S3 임베딩 설정 추가
    S3_CONFIG = {
        "bucket_name": "aws-medical-chatbot",
        "search_function": "medical-embedding-search",
        "enabled": True,  # 기본 활성화
        "cache_ttl_days": 7,  # 캐시 유효 기간
        "max_retries": 3  # API 실패 시 재시도 횟수
    }

    # 검색 소스 활성화 상태
    SEARCH_SOURCES = {
        "rag": False,     # 로컬 검색 기본 비활성화
        "s3": True,       # S3 검색 기본 활성화
        "medgemma": True, # MedGemma 검색 기본 활성화
        "pubmed": True    # PubMed 검색 기본 활성화
    }

    # 시스템 프롬프트들 (의료 특화)
    ROUTER_SYSTEM_PROMPT = """You are an expert at routing a medical question to a vectorstore or web search. 
    The vectorstore contains medical documents, clinical guidelines, emergency protocols, and treatment procedures. 
    
    Use the vectorstore for:
    - Medical procedures and protocols
    - Clinical guidelines and best practices  
    - Emergency response procedures
    - Drug information and interactions
    - Diagnostic criteria and methods
    
    Use web-search for:
    - Recent medical news and updates
    - Latest drug approvals or recalls
    - Current epidemic or health alerts
    - Real-time medical statistics
    
    Note that the user's question may be in Korean - this should not affect your routing decision."""
    
    GRADER_SYSTEM_PROMPT = """You are a medical information grader assessing relevance of a retrieved medical document to a healthcare question.
    
    Grade as relevant if the document contains:
    - Medical procedures, protocols, or guidelines related to the question
    - Clinical information about conditions, symptoms, or treatments mentioned
    - Emergency procedures or drug information relevant to the query
    - Diagnostic criteria or therapeutic approaches for the medical issue
    
    Focus on clinical relevance and patient safety. Give a binary score 'yes' or 'no' to indicate whether the document is medically relevant to the question.
    Note that the user's question may be in Korean - this is expected and you should assess relevance regardless of language."""
    
    RAG_SYSTEM_PROMPT = """You are a medical AI assistant specifically designed for healthcare professionals including doctors, nurses, and medical staff.
    
    Your role and responsibilities:
    - Provide accurate, evidence-based medical information
    - Support clinical decision-making with relevant guidelines
    - Offer emergency response protocols when applicable  
    - Include specific procedures, dosages, and protocols when relevant
    - Always prioritize patient safety in recommendations
    - Cite sources when available in the retrieved context
    
    Guidelines:
    - Use precise medical terminology appropriate for healthcare professionals
    - Include specific dosages, contraindications, and monitoring requirements when discussing medications
    - Provide step-by-step procedures for clinical interventions
    - Mention emergency protocols and when to escalate care
    - If uncertain about critical medical information, recommend consulting specialists
    - Always consider differential diagnoses and alternative approaches
    
    IMPORTANT: 
    - RESPOND IN THE SAME LANGUAGE AS THE USER'S INPUT (Korean for Korean input, English for English input, etc.)
    - Format responses professionally for medical staff
    - Include relevant medical disclaimers when appropriate
    - Structure information clearly with clinical priorities first"""
    
    HALLUCINATION_SYSTEM_PROMPT = """You are a medical information validator assessing whether an AI-generated medical response is grounded in the provided medical literature and clinical guidelines.
    
    Evaluation criteria:
    - Clinical accuracy and adherence to established medical standards
    - Proper citation of medical procedures and protocols
    - Accurate dosage information and contraindications
    - Appropriate emergency response recommendations
    - Consistency with current medical best practices
    
    Pay special attention to:
    - Drug dosages and administration routes
    - Emergency procedure sequences
    - Contraindications and precautions
    - Clinical decision-making pathways
    - Patient safety considerations
    
    Give a binary score 'yes' or 'no'. 'Yes' means the medical advice is grounded in established medical evidence and the provided context.
    RESPOND IN THE SAME LANGUAGE AS THE USER'S INPUT (Korean for Korean input, English for English input, etc.)"""
    
    REWRITER_SYSTEM_PROMPT = """You are a medical question re-writer that converts an input question to a better version optimized for medical document retrieval.
    
    Optimization strategies:
    - Add relevant medical terminology and synonyms
    - Include anatomical or physiological context when applicable
    - Expand abbreviations and medical acronyms
    - Add related symptoms, conditions, or procedures
    - Include department or specialty context when relevant
    
    Examples:
    - "심장이 아파요" → "흉통 심장통증 심근경색 협심증 심혈관질환 응급처치"
    - "당뇨 약" → "당뇨병 치료 약물 메트포민 인슐린 혈당조절 내분비내과"
    - "수술 후 관리" → "수술 후 처치 상처관리 합병증 예방 회복 간호"
    
    RESPOND IN THE SAME LANGUAGE AS THE USER'S INPUT (Korean for Korean input, English for English input, etc.), make it more comprehensive for medical document search."""

    @classmethod
    def get_medical_categories(cls) -> list:
        """의료 카테고리 목록 반환"""
        return cls.MEDICAL_CONFIG["categories"]
    
    @classmethod  
    def get_embedding_model(cls) -> str:
        """현재 임베딩 모델명 반환"""
        return cls.EMBEDDING_CONFIG["model"]
    
    @classmethod
    def is_openai_embedding(cls) -> bool:
        """OpenAI 임베딩 사용 여부"""
        return "text-embedding" in cls.EMBEDDING_CONFIG["model"]
    
    @classmethod
    def get_search_threshold(cls) -> float:
        """검색 유사도 임계값"""
        return cls.SIMILARITY_THRESHOLD
    
    @classmethod
    def validate_config(cls) -> Dict[str, Any]:
        """설정 유효성 검증"""
        issues = []
        
        # 필수 환경변수 확인
        import os
        if not os.getenv("OPENAI_API_KEY"):
            issues.append("OPENAI_API_KEY 환경변수가 설정되지 않음")
        
        # 모델명 검증
        if not cls.EMBEDDING_CONFIG["model"].startswith("text-embedding"):
            issues.append("지원되지 않는 임베딩 모델")
        
        # 임계값 범위 확인
        threshold = cls.SIMILARITY_THRESHOLD
        if not 0.0 <= threshold <= 1.0:
            issues.append("유사도 임계값이 범위를 벗어남 (0.0-1.0)")
        
        return {
            "valid": len(issues) == 0,
            "issues": issues,
            "config_type": "refactored_system",
            "embedding_model": cls.EMBEDDING_CONFIG["model"],
            "similarity_threshold": cls.SIMILARITY_THRESHOLD,
            "max_retries": cls.MAX_REWRITE_COUNT
        }
    
    @classmethod
    def update_system_prompt(cls, prompt_type: str, new_content: str) -> bool:
        """
        시스템 프롬프트 동적 업데이트
        
        Args:
            prompt_type: 프롬프트 타입 (예: 'RAG_SYSTEM_PROMPT')
            new_content: 새 프롬프트 내용
            
        Returns:
            성공 여부
        """
        try:
            if hasattr(cls, prompt_type) and isinstance(getattr(cls, prompt_type), str):
                setattr(cls, prompt_type, new_content)
                return True
            return False
        except Exception as e:
            print(f"프롬프트 업데이트 오류: {str(e)}")
            return False

    @classmethod
    def get_system_prompts(cls) -> Dict[str, str]:
        """
        모든 시스템 프롬프트 조회
        
        Returns:
            프롬프트 타입과 내용 딕셔너리
        """
        prompt_dict = {}
        for attr_name in dir(cls):
            if attr_name.endswith('_SYSTEM_PROMPT') and isinstance(getattr(cls, attr_name), str):
                prompt_dict[attr_name] = getattr(cls, attr_name)
        return prompt_dict