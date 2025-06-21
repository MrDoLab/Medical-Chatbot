# prompts.py
from typing import Dict, Any, Optional, List

class PromptTemplate:
    """프롬프트 템플릿 클래스"""
    
    def __init__(self, content: str, version: str = "1.0", description: str = "", variables: List[str] = None):
        """
        프롬프트 템플릿 초기화
        
        Args:
            content: 프롬프트 내용
            version: 프롬프트 버전
            description: 프롬프트 설명
            variables: 템플릿 변수 목록
        """
        self.content = content
        self.version = version
        self.description = description
        self.variables = variables or []
        
    def format(self, **kwargs) -> str:
        """
        템플릿 변수를 대체하여 프롬프트 생성
        
        Args:
            kwargs: 변수명과 값
            
        Returns:
            포맷된 프롬프트
        """
        formatted = self.content
        for var, value in kwargs.items():
            if var in self.variables:
                formatted = formatted.replace(f"{{{var}}}", str(value))
        return formatted

class SystemPrompts:
    """시스템 프롬프트 관리 클래스"""
    
    def __init__(self):
        """프롬프트 초기화"""
        self._prompts = {}
        self._initialize_prompts()
    
    def _initialize_prompts(self):
        """기본 프롬프트 초기화"""
        
        # 평가자 프롬프트
        self._prompts["GRADER"] = PromptTemplate(
            content="""You are a medical information grader assessing relevance of a retrieved medical document to a healthcare question.
            
            Grade as relevant if the document contains:
            - Medical procedures, protocols, or guidelines related to the question
            - Clinical information about conditions, symptoms, or treatments mentioned
            - Emergency procedures or drug information relevant to the query
            - Diagnostic criteria or therapeutic approaches for the medical issue
            
            Focus on clinical relevance and patient safety. Give a binary score 'yes' or 'no' to indicate whether the document is medically relevant to the question.
            Note that the user's question may be in Korean - this is expected and you should assess relevance regardless of language.""",
            version="1.0",
            description="검색된 문서의 의료 관련성 평가"
        )
        
        # RAG 프롬프트
        self._prompts["RAG"] = PromptTemplate(
            content="""You are a medical AI assistant specifically designed for healthcare professionals including doctors, nurses, and medical staff.
            
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
            - Structure information clearly with clinical priorities first""",
            version="1.0",
            description="의료 전문가용 RAG 응답 생성"
        )
        
        # 환각 평가 프롬프트
        self._prompts["HALLUCINATION"] = PromptTemplate(
            content="""You are a medical information validator assessing whether an AI-generated medical response is grounded in the provided medical literature and clinical guidelines.
            
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
            RESPOND IN THE SAME LANGUAGE AS THE USER'S INPUT (Korean for Korean input, English for English input, etc.)""",
            version="1.0",
            description="생성된 의료 정보의 환각 평가"
        )
        
        # 질문 재작성 프롬프트
        self._prompts["REWRITER"] = PromptTemplate(
            content="""You are a medical question re-writer that converts an input question to a better version optimized for medical document retrieval.
            
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
            
            RESPOND IN THE SAME LANGUAGE AS THE USER'S INPUT (Korean for Korean input, English for English input, etc.), make it more comprehensive for medical document search.""",
            version="1.0",
            description="의료 검색 최적화 질문 재작성"
        )
        
        # 통합기 프롬프트
        self._prompts["INTEGRATOR"] = PromptTemplate(
            content="""You are a medical information integrator. Combine multiple sources to provide accurate medical answers.

            Source Reliability Guide:
            - PubMed (Weight: {pubmed_weight}): Peer-reviewed academic papers - highest reliability
            - Bedrock KB (Weight: {bedrock_weight}): Curated medical knowledge base - very high reliability
            - RAG (Weight: {rag_weight}): Curated medical database - high reliability  
            - Web (Weight: {web_weight}): General web sources - moderate reliability
            - MedGemma (Weight: {medgemma_weight}): Medical language model - moderate reliability
            
            Integration Guidelines:
            - Prioritize information by source reliability
            - Synthesize complementary information from multiple sources
            - Note any important contradictions between sources
            - RESPOND IN THE SAME LANGUAGE AS THE USER'S INPUT (Korean for Korean input, English for English input, etc.)
            - Focus on medical accuracy and patient safety""",
            version="1.0",
            description="다중 소스 의료 정보 통합",
            variables=["pubmed_weight", "bedrock_weight", "rag_weight", "web_weight", "medgemma_weight"]
        )
        
        # 메모리 관리 프롬프트
        self._prompts["MEMORY"] = PromptTemplate(
            content="""You are a medical conversation summarizer. Create a concise summary of the conversation focusing on:

            1. Main medical topics discussed
            2. Key symptoms or conditions mentioned
            3. Important medical advice given
            4. Ongoing concerns or follow-up topics
            
            Keep the summary under 200 words and respond in {language}.""",
            version="1.0",
            description="의료 대화 요약 및 메모리 관리",
            variables=["language"]
        )
    
    def get(self, prompt_name: str) -> str:
        """
        프롬프트 내용 조회
        
        Args:
            prompt_name: 프롬프트 이름 (ROUTER, GRADER 등)
            
        Returns:
            프롬프트 내용
        """
        if prompt_name in self._prompts:
            return self._prompts[prompt_name].content
        return None
    
    def format(self, prompt_name: str, **kwargs) -> str:
        """
        템플릿 변수를 적용한 프롬프트 생성
        
        Args:
            prompt_name: 프롬프트 이름
            kwargs: 템플릿 변수
            
        Returns:
            포맷된 프롬프트
        """
        if prompt_name in self._prompts:
            return self._prompts[prompt_name].format(**kwargs)
        return None
    
    def update(self, prompt_name: str, content: str, version: str = None) -> bool:
        """
        프롬프트 내용 업데이트
        
        Args:
            prompt_name: 프롬프트 이름
            content: 새 프롬프트 내용
            version: 새 버전 (선택)
            
        Returns:
            성공 여부
        """
        if prompt_name in self._prompts:
            prompt = self._prompts[prompt_name]
            prompt.content = content
            if version:
                prompt.version = version
            return True
        return False
    
    def list_prompts(self) -> Dict[str, Dict[str, Any]]:
        """
        모든 프롬프트 정보 조회
        
        Returns:
            프롬프트 정보 딕셔너리
        """
        result = {}
        for name, prompt in self._prompts.items():
            result[name] = {
                "version": prompt.version,
                "description": prompt.description,
                "variables": prompt.variables,
                "preview": prompt.content[:100] + "..." if len(prompt.content) > 100 else prompt.content
            }
        return result
    
    def export_to_config_format(self) -> Dict[str, str]:
        """
        Config 클래스 형식으로 프롬프트 내보내기
        
        Returns:
            Config 클래스 형식의 프롬프트 딕셔너리
        """
        result = {}
        for name, prompt in self._prompts.items():
            result[f"{name}_SYSTEM_PROMPT"] = prompt.content
        return result
    
    def import_from_config(self, config_dict: Dict[str, str]) -> int:
        """
        Config 형식에서 프롬프트 가져오기
        
        Args:
            config_dict: Config 형식 프롬프트
            
        Returns:
            가져온 프롬프트 수
        """
        count = 0
        for key, value in config_dict.items():
            if key.endswith("_SYSTEM_PROMPT"):
                name = key.replace("_SYSTEM_PROMPT", "")
                if name in self._prompts:
                    self._prompts[name].content = value
                    count += 1
        return count

# 싱글톤 인스턴스 생성
system_prompts = SystemPrompts()