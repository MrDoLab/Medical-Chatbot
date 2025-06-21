from typing import List
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from prompts import system_prompts

class Generator:
    """답변 생성 담당 클래스"""
    
    def __init__(self, llm: ChatOpenAI):
        self.llm = llm
        self._setup_chains()
    
    def _setup_chains(self):
        """생성 체인 설정"""
        # RAG 답변 생성 체인
        self.rag_prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompts.get("RAG")),
            ("human", """Question: {question} \n\nContext: {context} \n\nAnswer:"""),
        ])
        self.rag_chain = self.rag_prompt | self.llm | StrOutputParser()
        
        # 질문 재작성 체인
        self.re_write_prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompts.get("REWRITER")),
            ("human", "Here is the initial question: \n\n {question} \n Formulate an improved question."),
        ])
        self.question_rewriter = self.re_write_prompt | self.llm | StrOutputParser()
    
    def format_docs(self, docs: List[Document]) -> str:
        """문서들을 형식화합니다."""
        return "\n\n".join([
            f'<document><content>{doc.page_content}</content><source>{doc.metadata.get("source", "unknown")}</source><page>{doc.metadata.get("page", 0)+1}</page></document>'
            for doc in docs
        ])
    
    def generate_answer(self, question: str, documents: List[Document]) -> str:
        """검색된 문서를 바탕으로 답변을 생성합니다."""
        print("==== [GENERATE] ====")
        
        # 문서가 없는 경우 기본 문서 추가
        if not documents or len(documents) == 0:
            documents = [
                Document(
                    page_content="인공지능 윤리는 AI 시스템의 개발과 사용에 관한 윤리적 지침을 포함합니다. 주요 원칙으로는 공정성, 투명성, 프라이버시, 책임성이 있습니다.",
                    metadata={"source": "fallback_document.txt", "page": 0}
                )
            ]
        
        generation = self.rag_chain.invoke({
            "context": self.format_docs(documents), 
            "question": question
        })
        return generation
    
    def rewrite_question(self, question: str) -> str:
        """쿼리를 재작성합니다."""
        print("==== [TRANSFORM QUERY] ====")
        better_question = self.question_rewriter.invoke({"question": question})
        print(f"원래 질문: {question}")
        print(f"재작성된 질문: {better_question}")
        return better_question