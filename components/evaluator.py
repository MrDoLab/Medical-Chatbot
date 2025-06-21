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
            ("human", "Set of facts: \n\n {documents} \n\n LLM generation: {generation}"),
        ])
        self.hallucination_grader = self.hallucination_prompt | self.hallucination_grader_llm
        self.answer_grader = self.hallucination_prompt | self.hallucination_grader_llm
    
    def grade_documents(self, question: str, documents: List[Document]) -> List[Document]:
        """검색된 문서의 관련성을 평가합니다."""
        print("==== [CHECK DOCUMENT RELEVANCE TO QUESTION] ====")
        filtered_docs = []
        for d in documents:
            score = self.retrieval_grader.invoke(
                {"question": question, "document": d.page_content}
            )
            grade = score.binary_score
            if grade == "yes":
                print("---GRADE: DOCUMENT RELEVANT---")
                filtered_docs.append(d)
            else:
                print("---GRADE: DOCUMENT NOT RELEVANT---")
                continue
        return filtered_docs
    
    def check_hallucination(self, documents: List[Document], generation: str, question: str) -> str:
        """생성된 답변의 할루시네이션 여부를 평가합니다."""
        print("==== [CHECK HALLUCINATIONS] ====")
        
        def format_docs(docs):
            return "\n\n".join([
                f'<document><content>{doc.page_content}</content><source>{doc.metadata.get("source", "unknown")}</source><page>{doc.metadata.get("page", 0)+1}</page></document>'
                for doc in docs
            ])
        
        # 환각 평가
        score = self.hallucination_grader.invoke(
            {"documents": format_docs(documents), "generation": generation}
        )
        grade = score.binary_score
        
        if grade == "yes":
            print("==== [DECISION: GENERATION IS GROUNDED IN DOCUMENTS] ====")
            # 답변의 관련성 평가
            print("==== [GRADE GENERATED ANSWER vs QUESTION] ====")
            score = self.answer_grader.invoke({"documents": documents, "generation": generation})
            grade = score.binary_score
            
            if grade == "yes":
                print("==== [DECISION: GENERATED ANSWER ADDRESSES QUESTION] ====")
                return "relevant"
            else:
                print("==== [DECISION: GENERATED ANSWER DOES NOT ADDRESS QUESTION] ====")
                return "not_relevant"
        else:
            print("==== [DECISION: GENERATION IS NOT GROUNDED IN DOCUMENTS, RE-TRY] ====")
            return "hallucination"

