from typing import List
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from prompts import system_prompts

class GradeDocuments(BaseModel):
    """Binary score for relevance check on retrieved documents."""
    binary_score: str = Field(
        description="Documents are relevant to the question, 'yes' or 'no'"
    )

class GradeHallucinations(BaseModel):
    """Binary score for hallucination present in generation answer."""
    binary_score: str = Field(
        description="Answer is grounded in the facts, 'yes' or 'no'"
    )

class Evaluator:
    """문서 및 답변 평가 담당 클래스"""
    
    def __init__(self, llm: ChatOpenAI):
        self.llm = llm
        self._setup_graders()
    
    def _setup_graders(self):
        """평가기 설정"""
        # 문서 관련성 평가기
        self.structured_llm_grader = self.llm.with_structured_output(GradeDocuments, method="function_calling")
        self.grade_prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompts.get("GRADER")),
            ("human", "Retrieved document: \n\n {document} \n\n User question: {question}"),
        ])
        self.retrieval_grader = self.grade_prompt | self.structured_llm_grader
        
        # 할루시네이션 평가기
        self.hallucination_grader_llm = self.llm.with_structured_output(GradeHallucinations, method="function_calling")
        self.hallucination_prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompts.get("HALLUCINATION")),
            ("human", "Set of facts: \n\n {documents} \n\n LLM generation: {generation} \n\n Question: {question}"),
        ])
        self.hallucination_grader = self.hallucination_prompt | self.hallucination_grader_llm
    
    def grade_documents(self, question: str, documents: List[Document]) -> List[Document]:
        """검색된 문서의 관련성을 평가합니다."""
        print("==== [CHECK DOCUMENT RELEVANCE TO QUESTION] ====")
        filtered_docs = []
        
        for d in documents:
            try:
                score = self.retrieval_grader.invoke(
                    {"question": question, "document": d.page_content}
                )
                grade = score.binary_score
                if grade.lower() == "yes":
                    print(f"---GRADE: DOCUMENT RELEVANT--- (Score: {d.metadata.get('similarity_score', 'N/A')})")
                    filtered_docs.append(d)
                else:
                    print(f"---GRADE: DOCUMENT NOT RELEVANT--- (Score: {d.metadata.get('similarity_score', 'N/A')})")
            except Exception as e:
                print(f"---ERROR GRADING DOCUMENT: {str(e)}---")
                # 오류 발생 시 일단 포함 (안전을 위해)
                filtered_docs.append(d)
                
        print(f"  📄 관련성 있는 문서: {len(filtered_docs)}/{len(documents)}개")
        return filtered_docs
    
    def check_hallucination(self, documents: List[Document], generation: str, question: str) -> str:
        """생성된 답변의 할루시네이션 여부를 평가합니다."""
        print("==== [CHECK HALLUCINATIONS] ====")
        
        if not documents:
            print("  ⚠️ 평가할 문서가 없습니다 - 환각 검사 생략")
            return "relevant"  # 문서가 없으면 검사 불가능하므로 통과시킴
        
        # 더 구조화된 문서 형식화
        formatted_docs = self._format_documents_for_evaluation(documents)
        
        try:
            # 환각 평가
            score = self.hallucination_grader.invoke({
                "documents": formatted_docs, 
                "generation": generation,
                "question": question
            })
            
            grade = score.binary_score.lower()
            
            if grade == "yes":
                print("==== [DECISION: Relevant] ====")
                return "relevant"
            else:
                print("==== [DECISION: Hallucination] ====")
                return "hallucination"
                
        except Exception as e:
            print(f"==== [HALLUCINATION CHECK ERROR: {str(e)}] ====")
            # 오류 발생 시 안전하게 처리
            print("  ⚠️ 환각 감지 실패 - 안전을 위해 재생성 진행")
            return "hallucination"
    
    def _format_documents_for_evaluation(self, documents: List[Document]) -> str:
        """평가용 문서 형식화 - 더 상세한 소스 정보 포함"""
        formatted_docs = []
        
        for i, doc in enumerate(documents):
            # 소스 정보 추출
            source_type = doc.metadata.get("source_type", "unknown")
            source = doc.metadata.get("source", "unknown")
            title = doc.metadata.get("title", "제목 없음")
            
            # 추가 메타데이터 (있는 경우)
            authors = doc.metadata.get("authors", "")
            if isinstance(authors, list):
                authors = ", ".join(authors)
            
            year = doc.metadata.get("year", "")
            journal = doc.metadata.get("journal", "")
            similarity = doc.metadata.get("similarity_score", "")
            
            # 문서 번호와 소스 타입으로 시작
            formatted_docs.append(f"--- DOCUMENT {i+1} [{source_type.upper()}] ---")
            formatted_docs.append(f"TITLE: {title}")
            
            # 추가 메타데이터 (존재하는 경우만)
            if authors:
                formatted_docs.append(f"AUTHORS: {authors}")
            if year:
                formatted_docs.append(f"YEAR: {year}")
            if journal:
                formatted_docs.append(f"JOURNAL: {journal}")
            if similarity:
                formatted_docs.append(f"RELEVANCE: {similarity:.4f}")
            
            formatted_docs.append(f"SOURCE: {source}")
            formatted_docs.append("CONTENT:")
            formatted_docs.append(doc.page_content)
            formatted_docs.append("---")
        
        return "\n".join(formatted_docs)