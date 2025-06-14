from typing import List, Dict, Any, Optional
from langchain_core.documents import Document
import requests
import xml.etree.ElementTree as ET
from datetime import datetime
import time

class PubMedSearcher:
    """PubMed 학술 논문 검색 담당 클래스"""
    
    def __init__(self, email: str = "user@example.com", api_key: str = None):
        self.email = email
        self.api_key = api_key
        self.base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
        self.search_url = f"{self.base_url}esearch.fcgi"
        self.fetch_url = f"{self.base_url}efetch.fcgi"
        
        # 검색 파라미터
        self.default_params = {
            "db": "pubmed",
            "tool": "medical_chatbot",
            "email": self.email,
            "retmode": "xml"
        }
        
        if self.api_key:
            self.default_params["api_key"] = self.api_key
    
    def search_pubmed(self, query: str, max_results: int = 5) -> List[Document]:
        """PubMed에서 관련 논문을 검색합니다"""
        print(f"==== [PUBMED SEARCH: {query}] ====")
        
        try:
            # 1단계: 논문 ID 검색
            pmids = self._search_paper_ids(query, max_results)
            
            if not pmids:
                print("  📄 PubMed 검색 결과 없음")
                return []
            
            print(f"  📄 {len(pmids)}개 논문 ID 발견")
            
            # 2단계: 논문 상세 정보 가져오기
            papers = self._fetch_paper_details(pmids)
            
            # 3단계: Document 객체로 변환
            documents = self._convert_to_documents(papers)
            
            print(f"  ✅ PubMed 검색 완료: {len(documents)}개 논문")
            return documents
            
        except Exception as e:
            print(f"  ❌ PubMed 검색 실패: {str(e)}")
            return self._create_fallback_documents(query)
    
    def _search_paper_ids(self, query: str, max_results: int) -> List[str]:
        """논문 ID들을 검색합니다"""
        # 의료 용어로 쿼리 최적화
        optimized_query = self._optimize_medical_query(query)
        
        search_params = self.default_params.copy()
        search_params.update({
            "term": optimized_query,
            "retmax": max_results,
            "sort": "relevance",
            "reldate": 1825,  # 최근 5년간 논문
        })
        
        try:
            response = requests.get(self.search_url, params=search_params, timeout=10)
            response.raise_for_status()
            
            # XML 파싱
            root = ET.fromstring(response.text)
            id_list = root.find("IdList")
            
            if id_list is not None:
                pmids = [id_elem.text for id_elem in id_list.findall("Id")]
                return pmids
            
            return []
            
        except Exception as e:
            print(f"    ❌ ID 검색 실패: {str(e)}")
            return []
    
    def _fetch_paper_details(self, pmids: List[str]) -> List[Dict]:
        """논문 상세 정보를 가져옵니다"""
        if not pmids:
            return []
        
        fetch_params = self.default_params.copy()
        fetch_params.update({
            "id": ",".join(pmids),
            "rettype": "abstract",
        })
        
        try:
            response = requests.get(self.fetch_url, params=fetch_params, timeout=15)
            response.raise_for_status()
            
            # XML 파싱
            root = ET.fromstring(response.text)
            papers = []
            
            for article in root.findall(".//PubmedArticle"):
                paper_info = self._parse_article(article)
                if paper_info:
                    papers.append(paper_info)
            
            return papers
            
        except Exception as e:
            print(f"    ❌ 논문 상세 정보 가져오기 실패: {str(e)}")
            return []
    
    def _parse_article(self, article) -> Optional[Dict]:
        """단일 논문 정보를 파싱합니다"""
        try:
            # 제목
            title_elem = article.find(".//ArticleTitle")
            title = title_elem.text if title_elem is not None else "제목 없음"
            
            # 초록
            abstract_elem = article.find(".//AbstractText")
            abstract = abstract_elem.text if abstract_elem is not None else "초록 없음"
            
            # 저자들
            authors = []
            for author in article.findall(".//Author"):
                last_name = author.find("LastName")
                first_name = author.find("ForeName")
                if last_name is not None and first_name is not None:
                    authors.append(f"{first_name.text} {last_name.text}")
            
            # 출판 연도
            pub_date = article.find(".//PubDate/Year")
            year = pub_date.text if pub_date is not None else "연도 미상"
            
            # 저널
            journal = article.find(".//Journal/Title")
            journal_name = journal.text if journal is not None else "저널 미상"
            
            # PMID
            pmid_elem = article.find(".//PMID")
            pmid = pmid_elem.text if pmid_elem is not None else "ID 없음"
            
            return {
                "title": title,
                "abstract": abstract,
                "authors": authors,
                "year": year,
                "journal": journal_name,
                "pmid": pmid
            }
            
        except Exception as e:
            print(f"    ❌ 논문 파싱 실패: {str(e)}")
            return None
    
    def _convert_to_documents(self, papers: List[Dict]) -> List[Document]:
        """논문 정보를 Document 객체로 변환합니다"""
        documents = []
        
        for paper in papers:
            # 논문 내용 구성
            content_parts = [
                f"제목: {paper['title']}",
                f"초록: {paper['abstract']}",
                f"저자: {', '.join(paper['authors'][:3])}{'...' if len(paper['authors']) > 3 else ''}",
                f"출판년도: {paper['year']}",
                f"저널: {paper['journal']}"
            ]
            
            content = "\n\n".join(content_parts)
            
            # 메타데이터 구성
            metadata = {
                "source": f"pubmed_{paper['pmid']}",
                "title": paper["title"],
                "authors": paper["authors"],
                "year": paper["year"],
                "journal": paper["journal"],
                "pmid": paper["pmid"],
                "source_type": "pubmed",
                "reliability": "high"  # PubMed는 기본적으로 높은 신뢰도
            }
            
            document = Document(page_content=content, metadata=metadata)
            documents.append(document)
        
        return documents
    
    def _optimize_medical_query(self, query: str) -> str:
        """의료 검색을 위한 쿼리 최적화"""
        # 한글 의료 용어를 영어로 변환
        medical_translations = {
            "낙상": "falls elderly",
            "당뇨병": "diabetes mellitus",  
            "고혈압": "hypertension",
            "골절": "fracture",
            "심장마비": "cardiac arrest myocardial infarction",
            "뇌졸중": "stroke cerebrovascular",
            "응급처치": "emergency treatment first aid",
            "호흡곤란": "dyspnea respiratory distress",
            "의식불명": "unconsciousness coma"
        }
        
        optimized_query = query.lower()
        
        # 한글 용어를 영어로 변환
        for korean, english in medical_translations.items():
            if korean in optimized_query:
                optimized_query = optimized_query.replace(korean, english)
        
        # PubMed 검색 최적화 키워드 추가
        if not any(word in optimized_query for word in ["treatment", "therapy", "management", "diagnosis"]):
            optimized_query += " treatment management"
        
        return optimized_query
    
    def _create_fallback_documents(self, query: str) -> List[Document]:
        """PubMed 검색 실패 시 기본 문서 생성"""
        fallback_content = f"""
        {query}에 관한 일반적인 의학 정보:
        
        이 정보는 PubMed 검색이 실패하여 제공되는 기본 정보입니다.
        정확한 의학적 정보는 의료 전문가와 상담하시기 바랍니다.
        
        일반적인 의료 권고사항:
        1. 증상이 지속되거나 악화되면 의료진과 상담
        2. 응급상황 시 119 신고
        3. 자가 진단보다는 전문의 진료 권장
        """
        
        return [Document(
            page_content=fallback_content,
            metadata={
                "source": "pubmed_fallback",
                "source_type": "pubmed",
                "reliability": "low",
                "note": "PubMed 검색 실패로 인한 기본 정보"
            }
        )]
