# components/retriever.py (완전한 리팩토링된 버전)
"""
문서 검색 전용 클래스 - 임베딩 생성 및 유사도 검색에 집중
문서 로딩은 DocumentLoader에게 위임
"""

import os
import json
import openai
import numpy as np
import pickle
import hashlib
from typing import List, Dict, Any, Optional
from pathlib import Path
from datetime import datetime, timedelta
from langchain_core.documents import Document
from components.parallel_searcher import ParallelSearcher
from components.document_loader import DocumentLoader
from config import Config
import logging

logger = logging.getLogger(__name__)

class Retriever:
    """리팩토링된 검색 전용 클래스"""
    
    def __init__(self):
        """검색기 초기화 - 검색 기능에만 집중"""
        
        # OpenAI 클라이언트 초기화
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY 환경변수가 설정되지 않았습니다!")
        
        self.client = openai.OpenAI(api_key=api_key)
        self.model_name = "text-embedding-3-large"
        
        # 문서 로더 (위임)
        self.document_loader = DocumentLoader()
        
        # 캐시 설정
        self.cache_dir = Path("./embedding_cache")
        self.cache_dir.mkdir(exist_ok=True)
        self.cache_enabled = True
        
        # 검색용 데이터
        self.medical_documents = []
        self.document_embeddings = []
        self.embedding_index = {}
        
        # 검색 통계
        self.search_stats = {
            "api_calls": 0,
            "cache_hits": 0,
            "total_tokens": 0,
            "searches_performed": 0,
            "average_response_time": 0.0
        }
        
        from components.pubMed_searcher import PubMedSearcher
        self.pubmed_searcher = PubMedSearcher(
            email="medical.chatbot@example.com",
            api_key=None
        )
        
        # 캐시 파일 경로
        self.embeddings_file = Path("./embeddings_cache.pkl")
        self.documents_file = Path("./documents_cache.pkl")
        
        print("🔍 검색기 초기화 중...")
        self._load_cached_embeddings()
        print(f"✅ 검색기 준비 완료! 현재 문서: {len(self.medical_documents)}개")
    
    def retrieve_documents(self, question: str, k: int = 5) -> List[Document]:
        """의료 질문에 대한 관련 문서 검색"""
        import time
        start_time = time.time()
        
        print(f"==== [SEARCH: {question[:50]}...] ====")
        
        try:
            # 1. 질문 임베딩 생성
            question_embedding = self._get_embedding(question)
            
            # 2. 의료 키워드 기반 사전 필터링
            candidate_indices = self._get_candidate_documents(question)
            
            # 3. 유사도 계산
            similarities = []
            indices_to_check = candidate_indices if candidate_indices else range(len(self.medical_documents))
            
            for i in indices_to_check:
                if i < len(self.document_embeddings):
                    similarity = self._cosine_similarity(question_embedding, self.document_embeddings[i])
                    similarities.append((similarity, i))
            
            # 4. 유사도순 정렬 및 임계값 적용
            similarities.sort(reverse=True)
            
            # 5. 상위 문서 선택
            top_documents = []
            threshold = getattr(Config, 'SIMILARITY_THRESHOLD', 0.3)
            
            for similarity, idx in similarities[:k*2]:
                if similarity >= threshold:
                    doc = self.medical_documents[idx].copy()
                    doc.metadata["similarity_score"] = round(similarity, 4)
                    doc.metadata["search_rank"] = len(top_documents) + 1
                    doc.metadata["search_question"] = question
                    top_documents.append(doc)
            
            # 6. 의료 관련성 재검증
            filtered_docs = self._medical_relevance_filter(top_documents, question)[:k]
            
            # 검색 통계 업데이트
            response_time = time.time() - start_time
            self._update_search_stats(response_time)
            
            print(f"  📊 검색 결과:")
            print(f"    후보: {len(indices_to_check)}개 → 유사: {len([s for s, _ in similarities if s >= threshold])}개 → 최종: {len(filtered_docs)}개")
            print(f"    응답시간: {response_time:.2f}초")
            if filtered_docs:
                print(f"    최고 유사도: {filtered_docs[0].metadata.get('similarity_score', 0):.3f}")
            
            return filtered_docs
            
        except Exception as e:
            logger.error(f"문서 검색 실패: {str(e)}")
            return self._get_emergency_fallback_docs(question)
    
    def load_documents_from_directory(self, directory_path: str) -> int:
        """문서 로딩 (DocumentLoader에게 위임)"""
        print(f"📚 문서 로딩 요청: {directory_path}")
        
        # 기존 문서 수
        initial_count = len(self.medical_documents)
        
        # DocumentLoader를 통해 문서 로딩
        new_documents = self.document_loader.load_documents_from_directory(directory_path)
        
        if not new_documents:
            print("📭 새로 로드할 문서가 없습니다")
            return 0
        
        # 중복 문서 체크
        existing_sources = {doc.metadata.get("source", "") for doc in self.medical_documents}
        unique_documents = []
        
        for doc in new_documents:
            source = doc.metadata.get("source", "")
            if source not in existing_sources:
                unique_documents.append(doc)
            else:
                print(f"  ⏭️ 중복 건너뜀: {Path(source).name}")
        
        if not unique_documents:
            print("📭 새로운 고유 문서가 없습니다 (모두 중복)")
            return 0
        
        # 새 문서들 임베딩 생성
        print(f"🧠 {len(unique_documents)}개 신규 문서 임베딩 생성 중...")
        
        texts_to_embed = [doc.page_content for doc in unique_documents]
        new_embeddings = self._batch_generate_embeddings(texts_to_embed)
        
        # 기존 데이터에 추가
        self.medical_documents.extend(unique_documents)
        self.document_embeddings.extend(new_embeddings)
        
        # 인덱스 재구축
        self._update_document_index()
        
        # 캐시 저장
        self._save_embeddings_cache()
        
        loaded_count = len(unique_documents)
        print(f"✅ 문서 로딩 완료: {loaded_count}개 신규, 총 {len(self.medical_documents)}개")
        
        return loaded_count
    
    def _get_embedding(self, text: str, use_cache: bool = True) -> List[float]:
        """텍스트를 OpenAI 임베딩으로 변환"""
        
        # 캐시 확인
        if use_cache and self.cache_enabled:
            cached = self._get_cached_embedding(text)
            if cached is not None:
                self.search_stats["cache_hits"] += 1
                return cached
        
        try:
            response = self.client.embeddings.create(
                model=self.model_name,
                input=text
            )
            
            embedding = response.data[0].embedding
            
            # 통계 업데이트
            self.search_stats["api_calls"] += 1
            self.search_stats["total_tokens"] += response.usage.total_tokens
            
            # 캐시 저장
            if use_cache and self.cache_enabled:
                self._save_cached_embedding(text, embedding)
            
            return embedding
            
        except Exception as e:
            logger.error(f"임베딩 생성 실패: {str(e)}")
            return [0.0] * 3072
    
    def _batch_generate_embeddings(self, texts: List[str], batch_size: int = 100) -> List[List[float]]:
        """여러 텍스트의 임베딩을 배치로 생성"""
        all_embeddings = []
        
        print(f"  🔄 {len(texts)}개 문서 임베딩 생성 중...")
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            batch_start = i + 1
            batch_end = min(i + batch_size, len(texts))
            
            print(f"    📦 배치 처리: {batch_start}-{batch_end}/{len(texts)}")
            
            try:
                response = self.client.embeddings.create(
                    model=self.model_name,
                    input=batch
                )
                
                batch_embeddings = [item.embedding for item in response.data]
                all_embeddings.extend(batch_embeddings)
                
                # 통계 업데이트
                self.search_stats["api_calls"] += 1
                self.search_stats["total_tokens"] += response.usage.total_tokens
                
            except Exception as e:
                logger.error(f"배치 임베딩 실패: {str(e)}")
                fallback_embeddings = [[0.0] * 3072] * len(batch)
                all_embeddings.extend(fallback_embeddings)
        
        return all_embeddings
    
    def _get_cached_embedding(self, text: str) -> Optional[List[float]]:
        """캐시된 임베딩 조회"""
        text_hash = hashlib.md5(text.encode()).hexdigest()
        cache_file = self.cache_dir / f"{text_hash}.pkl"
        
        if cache_file.exists():
            try:
                with open(cache_file, 'rb') as f:
                    cached_data = pickle.load(f)
                
                if datetime.now() - cached_data['timestamp'] < timedelta(days=7):
                    return cached_data['embedding']
            except:
                pass
        
        return None
    
    def _save_cached_embedding(self, text: str, embedding: List[float]):
        """임베딩을 캐시에 저장"""
        text_hash = hashlib.md5(text.encode()).hexdigest()
        cache_file = self.cache_dir / f"{text_hash}.pkl"
        
        try:
            cache_data = {
                'embedding': embedding,
                'timestamp': datetime.now(),
                'model': self.model_name,
                'text_preview': text[:100]
            }
            
            with open(cache_file, 'wb') as f:
                pickle.dump(cache_data, f)
        except Exception as e:
            logger.warning(f"캐시 저장 실패: {str(e)}")
    
    def _load_cached_embeddings(self):
        """저장된 임베딩과 문서 로드"""
        try:
            if self.embeddings_file.exists() and self.documents_file.exists():
                print("💾 기존 임베딩 데이터 로딩 중...")
                
                with open(self.documents_file, 'rb') as f:
                    self.medical_documents = pickle.load(f)
                
                with open(self.embeddings_file, 'rb') as f:
                    self.document_embeddings = pickle.load(f)
                
                self._update_document_index()
                
                print(f"✅ 기존 데이터 로드 완료: {len(self.medical_documents)}개 문서")
                return True
        except Exception as e:
            print(f"⚠️ 기존 데이터 로드 실패: {str(e)}")
        
        return False

    def _save_embeddings_cache(self):
        """임베딩과 문서를 파일로 저장"""
        try:
            print("💾 임베딩 데이터 저장 중...")
            
            with open(self.documents_file, 'wb') as f:
                pickle.dump(self.medical_documents, f)
            
            with open(self.embeddings_file, 'wb') as f:
                pickle.dump(self.document_embeddings, f)
            
            print("✅ 임베딩 데이터 저장 완료")
        except Exception as e:
            print(f"❌ 저장 실패: {str(e)}")
    
    def _get_candidate_documents(self, question: str) -> Optional[List[int]]:
        """질문에서 의료 키워드를 추출하여 후보 문서 필터링"""
        
        medical_keyword_map = {
            "낙상": ["낙상", "외상", "골절"],
            "당뇨": ["당뇨병", "혈당", "인슐린"],
            "고혈압": ["고혈압", "혈압", "심혈관"],
            "심정지": ["심정지", "CPR", "응급처치"],
            "응급": ["응급처치", "응급상황", "응급실"],
            "골절": ["골절", "외상", "정형외과"],
            "약물": ["약물", "처방", "부작용"],
            "수술": ["수술", "시술", "마취"]
        }
        
        question_lower = question.lower()
        candidate_indices = set()
        
        # 키워드 기반 후보 선정
        for keyword, related_terms in medical_keyword_map.items():
            if any(term in question_lower for term in related_terms):
                if keyword in self.embedding_index.get("keyword", {}):
                    candidate_indices.update(self.embedding_index["keyword"][keyword])
        
        # 후보가 너무 적으면 전체 검색
        if len(candidate_indices) < 10:
            return None
        
        return list(candidate_indices)
    
    def _cosine_similarity(self, a: List[float], b: List[float]) -> float:
        """코사인 유사도 계산"""
        a_np = np.array(a, dtype=np.float32)
        b_np = np.array(b, dtype=np.float32)
        
        norm_a = np.linalg.norm(a_np)
        norm_b = np.linalg.norm(b_np)
        
        if norm_a == 0 or norm_b == 0:
            return 0.0
        
        return float(np.dot(a_np, b_np) / (norm_a * norm_b))
    
    def _medical_relevance_filter(self, documents: List[Document], question: str) -> List[Document]:
        """의료 관련성 기반 최종 필터링"""
        
        scored_docs = []
        question_words = set(question.lower().split())
        
        for doc in documents:
            score = doc.metadata.get("similarity_score", 0)
            
            # 의료 관련성 보너스
            medical_bonus = 0
            
            # 1. 카테고리 보너스
            category = doc.metadata.get("category", "")
            if any(cat in category for cat in ["응급", "치료", "약물", "진단"]):
                medical_bonus += 0.1
            
            # 2. 키워드 매칭 보너스
            keywords = doc.metadata.get("keywords", [])
            keyword_matches = len([kw for kw in keywords if kw.lower() in question_words])
            medical_bonus += keyword_matches * 0.05
            
            # 3. 심각도 보너스
            severity = doc.metadata.get("severity", "medium")
            if severity == "critical" and any(word in question.lower() for word in ["응급", "급성", "즉시"]):
                medical_bonus += 0.15
            
            # 4. 신뢰도 보너스
            confidence = doc.metadata.get("confidence", "medium")
            if confidence == "high":
                medical_bonus += 0.05
            
            final_score = score + medical_bonus
            scored_docs.append((final_score, doc))
        
        scored_docs.sort(reverse=True)
        return [doc for score, doc in scored_docs]
    
    def _update_document_index(self):
        """검색 성능을 위한 인덱스 구축"""
        print("  🔍 검색 인덱스 구축 중...")
        
        category_index = {}
        keyword_index = {}
        
        for i, doc in enumerate(self.medical_documents):
            # 카테고리 인덱스
            category = doc.metadata.get("category", "기타")
            if category not in category_index:
                category_index[category] = []
            category_index[category].append(i)
            
            # 키워드 인덱스 (내용에서 추출)
            keywords = self._extract_keywords_from_content(doc.page_content)
            for keyword in keywords:
                if keyword not in keyword_index:
                    keyword_index[keyword] = []
                keyword_index[keyword].append(i)
        
        self.embedding_index = {
            "category": category_index,
            "keyword": keyword_index
        }
        
        print(f"    📊 카테고리: {len(category_index)}개")
        print(f"    🏷️ 키워드: {len(keyword_index)}개")
    
    def _extract_keywords_from_content(self, content: str) -> List[str]:
        """문서 내용에서 의료 키워드 추출"""
        medical_keywords = [
            "당뇨병", "고혈압", "심정지", "응급처치", "골절", "낙상",
            "약물", "처방", "부작용", "수술", "마취", "진단", "치료",
            "증상", "질환", "병원", "의사", "간호사", "환자"
        ]
        
        content_lower = content.lower()
        found_keywords = []
        
        for keyword in medical_keywords:
            if keyword in content_lower:
                found_keywords.append(keyword)
        
        return found_keywords
    
    def _update_search_stats(self, response_time: float):
        """검색 통계 업데이트"""
        self.search_stats["searches_performed"] += 1
        
        # 평균 응답시간 계산
        current_avg = self.search_stats["average_response_time"]
        search_count = self.search_stats["searches_performed"]
        
        new_avg = ((current_avg * (search_count - 1)) + response_time) / search_count
        self.search_stats["average_response_time"] = round(new_avg, 3)
    
    def _get_emergency_fallback_docs(self, question: str) -> List[Document]:
        """검색 실패 시 응급 폴백 문서"""
        return [
            Document(
                page_content=f"""
질문: {question}

현재 의료 지식 베이스에서 관련 문서를 찾을 수 없습니다.

일반적인 의료 응급상황 대응 원칙:
1. 환자 안전 최우선 확보
2. 생체징후 확인 (의식, 호흡, 맥박, 혈압)
3. 즉시 전문 의료진과 상담
4. 응급상황 시 119 신고

정확한 의료 정보를 위해서는 반드시 의료 전문가와 상담하시기 바랍니다.
""",
                metadata={
                    "source": "emergency_fallback",
                    "category": "응급안내",
                    "confidence": "low",
                    "search_question": question,
                    "fallback_type": "emergency"
                }
            )
        ]
    
    def _create_fallback_web_doc(self, question: str) -> List[Document]:
        """웹 검색 실패 시 폴백 문서"""
        return [
            Document(
                page_content=f"""
웹 검색 결과를 찾을 수 없습니다: {question}

의료 정보는 신뢰할 수 있는 출처에서 확인하시기 바랍니다:
- 의료진과 직접 상담
- 병원 공식 웹사이트
- 정부 보건 기관 자료

현재 로컬 의료 데이터베이스만 활용됩니다.
""",
                metadata={
                    "source": "web_fallback", 
                    "category": "검색실패",
                    "search_question": question,
                    "fallback_type": "web"
                }
            )
        ]
    
    def get_stats(self) -> Dict[str, Any]:
        """검색기 전체 통계"""
        cache_files = len(list(self.cache_dir.glob("*.pkl"))) if self.cache_enabled else 0
        
        loader_stats = self.document_loader.get_stats()
        
        return {
            "retriever_type": "RefactoredRetriever",
            "model_info": {
                "embedding_model": self.model_name,
                "dimensions": 3072
            },
            "document_stats": {
                "total_documents": len(self.medical_documents),
                "total_embeddings": len(self.document_embeddings),
                "index_categories": len(self.embedding_index.get("category", {})),
                "index_keywords": len(self.embedding_index.get("keyword", {}))
            },
            "search_performance": self.search_stats.copy(),
            "cache_info": {
                "cache_enabled": self.cache_enabled,
                "cache_files": cache_files,
                "cache_hit_rate": self.search_stats["cache_hits"] / max(1, self.search_stats["api_calls"] + self.search_stats["cache_hits"])
            },
            "cost_estimate": {
                "total_tokens": self.search_stats["total_tokens"],
                "estimated_cost_usd": self.search_stats["total_tokens"] * 0.13 / 1_000_000
            },
            "document_loader": loader_stats
        }
    
    def clear_cache(self):
        """캐시 초기화"""
        try:
            # 파일 캐시 삭제
            if self.embeddings_file.exists():
                self.embeddings_file.unlink()
            if self.documents_file.exists():
                self.documents_file.unlink()
            
            # 디렉토리 캐시 삭제
            for cache_file in self.cache_dir.glob("*.pkl"):
                cache_file.unlink()
            
            # 메모리 초기화
            self.medical_documents = []
            self.document_embeddings = []
            self.embedding_index = {}
            
            print("🗑️ 모든 캐시가 초기화되었습니다")
            
        except Exception as e:
            print(f"❌ 캐시 초기화 실패: {str(e)}")
    
    def reset_stats(self):
        """검색 통계 초기화"""
        self.search_stats = {
            "api_calls": 0,
            "cache_hits": 0,
            "total_tokens": 0,
            "searches_performed": 0,
            "average_response_time": 0.0
        }
        
        # DocumentLoader 통계도 초기화
        self.document_loader.reset_stats()
        
        print("📊 모든 통계가 초기화되었습니다")

# 테스트 및 사용 예시
def test_refactored_retriever():
    """리팩토링된 검색기 테스트"""
    print("🧪 리팩토링된 Retriever 테스트 시작\n")
    
    try:
        # 1. 검색기 초기화
        retriever = Retriever()
        
        # 2. 문서 로딩 테스트
        if Path("./medical_docs").exists():
            print("📚 문서 로딩 테스트...")
            count = retriever.load_documents_from_directory("./medical_docs")
            print(f"✅ {count}개 문서 로딩 완료\n")
        
        # 3. 검색 테스트
        test_queries = [
            "당뇨병 치료 방법",
            "응급처치 절차",
            "고혈압 약물"
        ]
        
        for query in test_queries:
            print(f"🔍 검색 테스트: '{query}'")
            docs = retriever.retrieve_documents(query, k=3)
            print(f"   결과: {len(docs)}개 문서")
            
            if docs:
                print(f"   최고 유사도: {docs[0].metadata.get('similarity_score', 0):.3f}")
            print()
        
        # 4. 통계 출력
        stats = retriever.get_stats()
        print("📊 검색기 통계:")
        print(f"   총 문서: {stats['document_stats']['total_documents']}개")
        print(f"   검색 횟수: {stats['search_performance']['searches_performed']}회")
        print(f"   평균 응답시간: {stats['search_performance']['average_response_time']}초")
        print(f"   캐시 적중률: {stats['cache_info']['cache_hit_rate']*100:.1f}%")
        
    except Exception as e:
        print(f"❌ 테스트 실패: {str(e)}")

if __name__ == "__main__":
    test_refactored_retriever()