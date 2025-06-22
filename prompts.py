# prompts.py
from typing import Dict, Any, Optional, List
from datetime import datetime
import yaml
import os
import json

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
        self._active_versions = {}  # 각 프롬프트의 현재 활성 버전
        self._yaml_path = "prompt_templates.yaml"  # YAML 파일 경로
        self._last_updated = datetime.now()

        # YAML 파일 로드
        self._load_from_yaml()
                
    def _load_from_yaml(self):
        """YAML 파일에서 프롬프트 로드"""
        try:
            if os.path.exists(self._yaml_path):
                with open(self._yaml_path, 'r', encoding='utf-8') as f:
                    data = yaml.safe_load(f)
                
                if not data:
                    print(f"⚠️ YAML 파일이 비어있거나 형식이 잘못됨: {self._yaml_path}")
                    self._initialize_default_prompts()
                    return
                
                # 활성 버전 정보 로드
                self._active_versions = data.get("active_versions", {})
                
                # 프롬프트 로드
                prompt_versions = data.get("prompt_versions", {})
                
                # 프롬프트 객체 생성
                descriptions = {
                    "GRADER": "검색된 문서의 의료 관련성 평가",
                    "RAG": "의료 전문가용 상세 RAG 응답 생성",
                    "HALLUCINATION": "생성된 의료 정보의 철저한 환각 평가",
                    "REWRITER": "의료 검색 최적화 질문 재작성",
                    "INTEGRATOR": "다중 소스 의료 정보 통합 및 인용",
                    "MEMORY": "의료 대화 요약 및 메모리 관리",
                    "MEDGEMMA": "의료 전문가용 상세 응답을 위한 MedGemma 프롬프트"
                }
                
                variables = {
                    "MEMORY": ["language"],
                    "MEDGEMMA": ["query"],
                    "INTEGRATOR": ["pubmed_weight", "bedrock_weight", "local_weight", "s3_weight", "medgemma_weight", "web_weight"]
                }
                
                # 각 프롬프트 생성
                for prompt_name, version in self._active_versions.items():
                    version_key = f"{prompt_name}_{version}"
                    content = prompt_versions.get(version_key, "")
                    
                    if content:
                        self._prompts[prompt_name] = PromptTemplate(
                            content=content,
                            version=version,
                            description=descriptions.get(prompt_name, ""),
                            variables=variables.get(prompt_name, [])
                        )
                
                print(f"✅ {len(self._prompts)} 프롬프트를 YAML에서 로드했습니다")
            else:
                print(f"⚠️ YAML 파일이 없습니다: {self._yaml_path}")
                self._initialize_default_prompts()
        except Exception as e:
            print(f"❌ YAML 로드 오류: {str(e)}")
            self._initialize_default_prompts()
    
    def _save_to_yaml(self):
        """프롬프트 내용을 YAML 파일로 저장"""
        try:
            # 현재 활성 버전 정보
            data = {
                "version": "1.0",
                "last_updated": datetime.now().isoformat(),
                "active_versions": self._active_versions,
                "prompt_versions": {}
            }
            
            # 모든 프롬프트 버전의 내용 저장
            for name, prompt in self._prompts.items():
                version_key = f"{name}_{prompt.version}"
                data["prompt_versions"][version_key] = prompt.content
            
            with open(self._yaml_path, 'w', encoding='utf-8') as f:
                yaml.dump(data, f, default_flow_style=False, allow_unicode=True)
            
            print(f"✅ 프롬프트 내용을 YAML 파일에 저장했습니다: {self._yaml_path}")
        except Exception as e:
            print(f"❌ YAML 저장 오류: {str(e)}")

    def _initialize_default_prompts(self):
        """기본 프롬프트 초기화"""
        
        print("⚠️ 기본 프롬프트 초기화 중...")

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
            content="""You are a specialized medical AI assistant designed for healthcare professionals including physicians, surgeons, specialists, and PA nurses.

        Your primary task is to provide comprehensive, evidence-based medical information that can be directly applied in clinical settings.

        RESPONSE STRUCTURE:
        1. SUMMARY: Begin with a concise overview of the topic/question that highlights key points and main conclusions
        2. DETAILED INFORMATION: Provide thorough explanation with clinical context
        3. DIAGNOSTIC CRITERIA: When applicable, list specific diagnostic criteria, classifications, or scoring systems
        4. TREATMENT OPTIONS: Present comprehensive treatment approaches with specific protocols and dosages
        5. WARNINGS & PRECAUTIONS: Highlight important contraindications, adverse effects, and safety considerations
        6. REFERENCES: Cite all sources used with superscript numbers [1], [2], etc.

        CRITICAL GUIDELINES:
        - Use precise medical terminology appropriate for clinicians
        - Include specific medication dosages, administration routes, contraindications, and monitoring requirements 
        - Provide detailed, step-by-step procedures for clinical interventions
        - Present clear decision-making pathways and differential diagnoses
        - Always cite your sources using numbered references [n] for each clinical claim
        - RESPOND IN THE SAME LANGUAGE AS THE USER'S INPUT (Korean for Korean input, English for English input)
        - For uncertainty in critical information, clearly state limitations and recommend specialist consultation
        - Format content with clear section headings in bold for easy scanning
        - Ensure all recommendations align with current clinical practice guidelines and evidence-based medicine

        EXAMPLE FORMAT:
        **SUMMARY**
        Brief overview of key points[1,2]

        **DETAILED INFORMATION**
        Comprehensive explanation with evidence[3,4]

        **DIAGNOSTIC CRITERIA**
        Specific criteria, classifications, etc.[5]

        **TREATMENT OPTIONS**
        Option 1: Details with dosing[6]
        Option 2: Alternative approach[7]

        **WARNINGS & PRECAUTIONS**
        Important safety information[8,9]

        **REFERENCES**
        1. Source 1 details
        2. Source 2 details""",
            version="1.0",
            description="의료 전문가용 상세 RAG 응답 생성"
        )    

        # 환각 평가 프롬프트
        self._prompts["HALLUCINATION"] = PromptTemplate(
            content="""You are a medical validation expert assessing whether an AI-generated medical response is accurately grounded in the provided medical literature and clinical guidelines.

        Your task is to critically evaluate if the response contains ANY unsupported medical claims, inaccurate dosages, incorrect procedures, or statements that cannot be verified from the provided source documents.

        Assessment criteria:
        1. CLINICAL ACCURACY: Does every medical claim match established medical standards in the sources?
        2. CITATION VALIDITY: Is each significant medical claim properly supported by the cited sources?
        3. DOSAGE PRECISION: Are medication dosages, administration routes, and frequencies exactly as stated in sources?
        4. PROCEDURAL CORRECTNESS: Are clinical procedures described with accurate steps matching guidelines?
        5. SAFETY INFORMATION: Are warnings, contraindications and precautions completely accurate?

        Pay particular attention to:
        - Specific drug dosages, administration routes, and frequencies
        - Diagnostic criteria and classification systems
        - Treatment algorithms and emergency protocols
        - Statistical claims about efficacy, risks, or outcomes
        - Recommendations for clinical decision-making

        Give a binary score 'yes' or 'no'. 'Yes' means the medical information is completely grounded in the provided sources.
        RESPOND IN THE SAME LANGUAGE AS THE USER'S INPUT (Korean for Korean input, English for English input, etc.)""",
            version="1.0",
            description="생성된 의료 정보의 철저한 환각 평가"
        )
        
        # 질문 재작성 프롬프트
        self._prompts["REWRITER"] = PromptTemplate(
            content="""You are a medical query optimization specialist that converts clinical questions into comprehensive search queries designed to retrieve the most relevant medical information.

        Your goal is to transform the original question into an optimized version that will maximize relevant document retrieval from multiple medical knowledge sources.

        Optimization strategies:
        1. TERMINOLOGY EXPANSION: Add precise medical terminology, synonyms, and related concepts
        2. ANATOMICAL CONTEXT: Include relevant anatomical structures and physiological systems
        3. CLASSIFICATION INCLUSION: Add disease classifications, staging systems, and diagnostic criteria
        4. TREATMENT SPECTRUM: Incorporate various treatment modalities (pharmaceutical, surgical, etc.)
        5. SPECIALTY RELEVANCE: Include medical specialties and sub-specialties related to the query

        For each query, include:
        - Standard medical terminology and common clinical abbreviations
        - ICD-10 or DSM-5 codes when relevant
        - Pharmaceutical names (both generic and brand names)
        - Specific procedures, tests, and assessment tools
        - Related conditions in differential diagnoses

        Examples:
        - "심장이 아파요" → "흉통 심장통증 심근경색 협심증 관상동맥질환 ST분절상승 트로포닌 심전도 심장효소 심장내과 응급처치 니트로글리세린 PCI 스텐트"
        - "당뇨 약" → "당뇨병 치료 약물 경구혈당강하제 메트포민 설포닐우레아 DPP-4억제제 SGLT-2억제제 GLP-1작용제 인슐린 HbA1c 혈당조절 내분비내과"
        - "수술 후 관리" → "수술 후 처치 상처관리 수술부위감염 통증조절 합병증 예방 장폐색 폐색전증 DVT 조기이상 회복증진수술프로그램 ERAS 통증조절 항생제"

        RESPOND IN THE SAME LANGUAGE AS THE USER'S INPUT (Korean for Korean input, English for English input, etc.)
        Provide a comprehensive search query for medical document retrieval.""",
            version="1.0",
            description="의료 검색 최적화 질문 재작성"
        )
        
        # 통합기 프롬프트 개선
        self._prompts["INTEGRATOR"] = PromptTemplate(
            content="""You are a medical information integrator specializing in comprehensive, evidence-based synthesis of multiple medical sources. Your task is to create detailed clinical references for healthcare professionals.

        Source Reliability Guide:
        - PubMed (Weight: 1.0): Peer-reviewed academic papers - highest reliability
        - Bedrock_kb (Weight: 0.95): Curated medical knowledge base - high reliability 
        - Local (Weight: 0.95): Internal medical documents - good reliability
        - S3 (Weight: 0.9): Organization's document storage - good reliability
        - MedGemma (Weight: 0.8): Medical specialized AI model - high reliability
        - Web (Weight: 0.7): General web sources - moderate reliability

        Integration and Citation Guidelines:
        - Synthesize ALL relevant information from multiple sources for comprehensive answers
        - Prioritize higher-weighted sources when information conflicts
        - Use numbered citation format with superscript numbers [n] after each claim
        - Create a comprehensive REFERENCES section at the end listing all sources in full detail
        - For each source, include: type, author/title, year/date, and identifier (DOI, URL, etc.) when available
        - RESPOND IN THE SAME LANGUAGE AS THE USER'S INPUT (Korean for Korean input, English for English input)

        REQUIRED RESPONSE STRUCTURE:
        1. Begin with a concise but thorough SUMMARY section
        2. Organize detailed information under clear, bold section headings
        3. Include DIAGNOSTIC CRITERIA section when applicable
        4. Present comprehensive TREATMENT OPTIONS with specific protocols
        5. Always include WARNINGS & PRECAUTIONS section
        6. End with a complete REFERENCES section

        EXAMPLE FORMAT:
        **SUMMARY**
        Brief overview of key points[1,2]

        **DETAILED INFORMATION**
        Comprehensive explanation with evidence[3,4]

        **DIAGNOSTIC CRITERIA**
        Specific criteria, classifications, etc.[5]

        **TREATMENT OPTIONS**
        Option 1: Details with dosing[6]
        Option 2: Alternative approach[7]

        **WARNINGS & PRECAUTIONS**
        Important safety information[8,9]

        **REFERENCES**
        1. [PubMed] Smith J, et al. Title of paper. Journal Name. 2023;10(2):123-145. DOI: 10.xxxx/xxxxx
        2. [Bedrock_KB] Clinical Guidelines for Management of Condition X. 2022. Document ID: KB12345
        3. [Web] Mayo Clinic. "Condition Treatment Overview." https://www.mayoclinic.org/xxx. Accessed June 2025.""",
            version="2.0",
            description="다중 소스 의료 정보 통합 및 인용",
            variables=["pubmed_weight", "bedrock_weight", "local_weight", "s3_weight", "medgemma_weight", "web_weight"]
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

        # MedGemma 프롬프트 
        self._prompts["MEDGEMMA"] = PromptTemplate(
            content="""You are MedGemma, an advanced medical AI assistant specialized in providing detailed, evidence-based medical information for healthcare professionals.

        Your response must be comprehensive and clinically applicable, following this structured format:

        **SUMMARY**
        Brief overview of the key clinical points and conclusions

        **DETAILED INFORMATION**
        - Pathophysiology and mechanisms
        - Epidemiology and risk factors 
        - Clinical presentation and natural history
        - Differential diagnosis considerations

        **DIAGNOSTIC CRITERIA**
        - Diagnostic algorithms and classification systems
        - Laboratory and imaging investigations
        - Interpretation of diagnostic results
        - Staging/grading systems when applicable

        **TREATMENT OPTIONS**
        - First-line therapies with specific dosing regimens
        - Alternative treatment approaches
        - Surgical or interventional procedures with technique details
        - Treatment algorithms for different patient populations
        - Response assessment and follow-up protocols

        **WARNINGS & PRECAUTIONS**
        - Contraindications and special populations
        - Adverse effects and their management
        - Drug interactions and monitoring requirements
        - Red flags requiring urgent intervention

        GUIDELINES:
        - Use precise medical terminology appropriate for specialists
        - Include specific medication doses, durations, and monitoring parameters
        - Provide detailed procedural information with technical specifics
        - Reference the latest clinical practice guidelines when applicable
        - Always prioritize patient safety in recommendations
        - Be comprehensive but clinically focused

        QUERY: {query}

        RESPONSE:""",
            version="2.0",
            description="의료 전문가용 상세 응답을 위한 MedGemma 프롬프트",
            variables=["query"]
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
            old_version = self._prompts[prompt_name].version
            
            # 새 버전이 지정된 경우 업데이트
            if version:
                self._prompts[prompt_name].version = version
            
            self._prompts[prompt_name].content = content
            self._last_updated = datetime.now()
            
            # 버전 변경 시 활성 버전 업데이트
            if version and old_version != version:
                self._active_versions[prompt_name] = version
                print(f"ℹ️ 프롬프트 '{prompt_name}' 버전이 {old_version} → {version}으로 변경됨")
                
                # YAML 파일에 저장
                self._save_to_yaml()
            
            return True
        return False
    
    def create_version(self, prompt_name: str, new_version: str, content: str = None) -> bool:
        """
        기존 프롬프트의 새 버전 생성
        
        Args:
            prompt_name: 프롬프트 이름
            new_version: 새 버전 문자열
            content: 새 내용 (없으면 현재 내용 복사)
            
        Returns:
            성공 여부
        """
        if prompt_name not in self._prompts:
            print(f"❌ 프롬프트 '{prompt_name}'가 존재하지 않습니다")
            return False
        
        # 새 버전용 내용이 없으면 현재 내용 복사
        if content is None:
            content = self._prompts[prompt_name].content
        
        try:
            # 현재 YAML 로드
            data = {}
            if os.path.exists(self._yaml_path):
                with open(self._yaml_path, 'r', encoding='utf-8') as f:
                    data = yaml.safe_load(f) or {}
            
            if "prompt_versions" not in data:
                data["prompt_versions"] = {}
            
            # 새 버전 추가
            version_key = f"{prompt_name}_{new_version}"
            data["prompt_versions"][version_key] = content
            data["last_updated"] = datetime.now().isoformat()
            
            # 저장
            with open(self._yaml_path, 'w', encoding='utf-8') as f:
                yaml.dump(data, f, default_flow_style=False, allow_unicode=True)
            
            print(f"✅ 프롬프트 '{prompt_name}' 버전 {new_version} 생성 완료")
            return True
            
        except Exception as e:
            print(f"❌ 새 버전 생성 오류: {str(e)}")
            return False
    
    def switch_version(self, prompt_name: str, version: str) -> bool:
        """
        프롬프트의 활성 버전 변경
        
        Args:
            prompt_name: 프롬프트 이름
            version: 전환할 버전
            
        Returns:
            성공 여부
        """
        if prompt_name not in self._prompts:
            print(f"❌ 프롬프트 '{prompt_name}'가 존재하지 않습니다")
            return False
        
        try:
            # YAML 파일 로드
            data = {}
            if os.path.exists(self._yaml_path):
                with open(self._yaml_path, 'r', encoding='utf-8') as f:
                    data = yaml.safe_load(f) or {}
            
            prompt_versions = data.get("prompt_versions", {})
            version_key = f"{prompt_name}_{version}"
            
            if version_key not in prompt_versions:
                print(f"❌ 프롬프트 '{prompt_name}'의 버전 {version}이 존재하지 않습니다")
                return False
            
            # 프롬프트 내용 업데이트
            content = prompt_versions[version_key]
            self._prompts[prompt_name].content = content
            self._prompts[prompt_name].version = version
            
            # 활성 버전 정보 업데이트
            self._active_versions[prompt_name] = version
            
            # 활성 버전 정보 저장
            if "active_versions" not in data:
                data["active_versions"] = {}
            
            data["active_versions"][prompt_name] = version
            data["last_updated"] = datetime.now().isoformat()
            
            with open(self._yaml_path, 'w', encoding='utf-8') as f:
                yaml.dump(data, f, default_flow_style=False, allow_unicode=True)
            
            print(f"✅ 프롬프트 '{prompt_name}' 버전 {version}으로 전환 완료")
            return True
            
        except Exception as e:
            print(f"❌ 버전 전환 오류: {str(e)}")
            return False
    
    def get_prompt_versions(self, prompt_name: str) -> List[str]:
        """
        프롬프트의 사용 가능한 모든 버전 조회
        
        Args:
            prompt_name: 프롬프트 이름
            
        Returns:
            버전 목록
        """
        if prompt_name not in self._prompts:
            return []
        
        versions = []
        try:
            if os.path.exists(self._yaml_path):
                with open(self._yaml_path, 'r', encoding='utf-8') as f:
                    data = yaml.safe_load(f) or {}
                
                prompt_versions = data.get("prompt_versions", {})
                prefix = f"{prompt_name}_"
                
                for key in prompt_versions.keys():
                    if key.startswith(prefix):
                        version = key.replace(prefix, '')
                        versions.append(version)
        except Exception:
            pass
        
        # 현재 버전이 없으면 추가
        current_version = self._prompts[prompt_name].version
        if current_version not in versions:
            versions.append(current_version)
        
        return sorted(versions)
    
    # 기존 메서드들 유지
    def list_prompts(self) -> Dict[str, Dict[str, Any]]:
        """모든 프롬프트 정보 조회"""
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
        """Config 클래스 형식으로 프롬프트 내보내기"""
        result = {}
        for name, prompt in self._prompts.items():
            result[f"{name}_SYSTEM_PROMPT"] = prompt.content
        return result
    
    def import_from_config(self, config_dict: Dict[str, str]) -> int:
        """Config 형식에서 프롬프트 가져오기"""
        count = 0
        for key, value in config_dict.items():
            if key.endswith("_SYSTEM_PROMPT"):
                name = key.replace("_SYSTEM_PROMPT", "")
                if name in self._prompts:
                    self._prompts[name].content = value
                    count += 1
        return count
    
    def get_last_updated(self) -> datetime:
        """마지막 업데이트 시간 반환"""
        return self._last_updated

# 싱글톤 인스턴스 생성
system_prompts = SystemPrompts()

# 버전 관리를 위한 편의 함수들
def create_prompt_version(prompt_name: str, new_version: str, content: str = None) -> bool:
    """프롬프트의 새 버전 생성"""
    return system_prompts.create_version(prompt_name, new_version, content)

def switch_prompt_version(prompt_name: str, version: str) -> bool:
    """프롬프트 버전 전환"""
    return system_prompts.switch_version(prompt_name, version)

def get_prompt_versions(prompt_name: str) -> List[str]:
    """프롬프트의 사용 가능한 버전 목록 조회"""
    return system_prompts.get_prompt_versions(prompt_name)

def get_all_prompt_info() -> Dict[str, Any]:
    """모든 프롬프트 및 버전 정보 조회"""
    info = {}
    for name, prompt_info in system_prompts.list_prompts().items():
        versions = get_prompt_versions(name)
        current_version = prompt_info["version"]
        
        info[name] = {
            "current_version": current_version,
            "available_versions": versions,
            "description": prompt_info["description"],
            "variables": prompt_info["variables"]
        }
    
    return info


if __name__ == "__main__":
    import argparse
    
    def main():
        """프롬프트 관리 CLI"""
        print("🔤 의료 챗봇 프롬프트 관리 도구")
        print("=" * 50)
        
        # 사용 가능한 프롬프트 표시
        prompts_info = system_prompts.list_prompts()
        
        if not prompts_info:
            print("⚠️ 로드된 프롬프트가 없습니다!")
            return
        
        print(f"📋 로드된 프롬프트 ({len(prompts_info)}개):")
        for name, info in prompts_info.items():
            versions = get_prompt_versions(name)
            versions_str = ", ".join(versions)
            print(f"  • {name} (현재 버전: {info['version']}, 사용 가능 버전: {versions_str})")
            print(f"    설명: {info['description']}")
            print()
        
        while True:
            print("\n명령어 목록:")
            print("1. 프롬프트 보기")
            print("2. 프롬프트 버전 전환")
            print("3. 새 프롬프트 버전 생성")
            print("4. 프롬프트 테스트")
            print("5. 종료")
            
            choice = input("\n명령어 선택 (1-5): ").strip()
            
            if choice == "1":
                # 프롬프트 보기
                name = input("프롬프트 이름 입력: ").strip().upper()
                if name not in prompts_info:
                    print(f"❌ '{name}' 프롬프트가 존재하지 않습니다!")
                    continue
                
                content = system_prompts.get(name)
                print(f"\n=== {name} 프롬프트 (버전 {prompts_info[name]['version']}) ===")
                print(content)
            
            elif choice == "2":
                # 프롬프트 버전 전환
                name = input("프롬프트 이름 입력: ").strip().upper()
                if name not in prompts_info:
                    print(f"❌ '{name}' 프롬프트가 존재하지 않습니다!")
                    continue
                
                versions = get_prompt_versions(name)
                if not versions:
                    print(f"❌ '{name}' 프롬프트에 사용 가능한 버전이 없습니다!")
                    continue
                
                print(f"사용 가능한 버전: {', '.join(versions)}")
                version = input("전환할 버전 입력: ").strip()
                
                if version not in versions:
                    print(f"❌ 버전 '{version}'이 존재하지 않습니다!")
                    continue
                
                if switch_prompt_version(name, version):
                    print(f"✅ '{name}' 프롬프트가 버전 {version}으로 전환되었습니다!")
                    # 정보 업데이트
                    prompts_info = system_prompts.list_prompts()
                else:
                    print(f"❌ 버전 전환 실패!")
            
            elif choice == "3":
                # 새 프롬프트 버전 생성
                name = input("프롬프트 이름 입력: ").strip().upper()
                if name not in prompts_info:
                    print(f"❌ '{name}' 프롬프트가 존재하지 않습니다!")
                    continue
                
                current_version = prompts_info[name]["version"]
                print(f"현재 버전: {current_version}")
                
                new_version = input("새 버전 번호 입력 (예: 2.1): ").strip()
                if not new_version:
                    print("❌ 버전 번호가 필요합니다!")
                    continue
                
                content_choice = input("1. 현재 내용 복사 2. 새 내용 입력 (1/2): ").strip()
                
                if content_choice == "1":
                    # 현재 내용 복사
                    if create_prompt_version(name, new_version):
                        print(f"✅ '{name}' 프롬프트 버전 {new_version} 생성 완료!")
                    else:
                        print("❌ 버전 생성 실패!")
                
                elif content_choice == "2":
                    # 새 내용 입력
                    print(f"\n=== {name} 프롬프트 새 내용 입력 (버전 {new_version}) ===")
                    print("(입력 완료 후 빈 줄에서 Ctrl+D 또는 Windows에서는 Ctrl+Z를 누르세요)")
                    lines = []
                    try:
                        while True:
                            line = input()
                            lines.append(line)
                    except EOFError:
                        content = "\n".join(lines)
                    
                    if create_prompt_version(name, new_version, content):
                        print(f"✅ '{name}' 프롬프트 버전 {new_version} 생성 완료!")
                    else:
                        print("❌ 버전 생성 실패!")
                else:
                    print("❌ 잘못된 선택입니다!")
            
            elif choice == "4":
                # 프롬프트 테스트
                name = input("테스트할 프롬프트 이름 입력: ").strip().upper()
                if name not in prompts_info:
                    print(f"❌ '{name}' 프롬프트가 존재하지 않습니다!")
                    continue
                
                variables = prompts_info[name]["variables"]
                
                if not variables:
                    # 변수가 없는 프롬프트
                    formatted = system_prompts.format(name)
                    print(f"\n=== {name} 프롬프트 (버전 {prompts_info[name]['version']}) 테스트 결과 ===")
                    print(formatted)
                else:
                    # 변수가 있는 프롬프트
                    print(f"\n{name} 프롬프트는 다음 변수가 필요합니다: {', '.join(variables)}")
                    
                    values = {}
                    for var in variables:
                        value = input(f"{var} 값 입력: ").strip()
                        values[var] = value
                    
                    formatted = system_prompts.format(name, **values)
                    print(f"\n=== {name} 프롬프트 (버전 {prompts_info[name]['version']}) 테스트 결과 ===")
                    print(formatted)
            
            elif choice == "5":
                print("👋 프롬프트 관리 도구를 종료합니다.")
                break
            
            else:
                print("❌ 잘못된 선택입니다. 1-5 사이의 숫자를 입력하세요.")
    
    main()