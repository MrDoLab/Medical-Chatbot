from typing import Literal
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from config import Config

class RouteQuery(BaseModel):
    """Route a user query to the most relevant datasource."""
    datasource: Literal["vectorstore", "web_search"] = Field(
        ...,
        description="Given a user question choose to route it to web search or a vectorstore.",
    )

class Router:
    """질문 라우팅 담당 클래스"""
    
    def __init__(self, llm: ChatOpenAI):
        self.llm = llm
        self.structured_llm_router = llm.with_structured_output(RouteQuery)
        self.route_prompt = ChatPromptTemplate.from_messages([
            ("system", Config.ROUTER_SYSTEM_PROMPT),
            ("human", "{question}"),
        ])
        self.question_router = self.route_prompt | self.structured_llm_router
    
    def route_question(self, question: str) -> str:
        return "vectorstore"