# components/text_processor.py
"""
텍스트 파일 전용 처리기 - .txt, .md, .json 등
"""

import json
import re
from typing import Dict, List
from pathlib import Path
from langchain_core.documents import Document
from datetime import datetime

class TextProcessor:
    """텍스트 파일 전용 처리기"""
    
    def __init__(self):
        """텍스트 처리기 초기화"""
        self.max_content_length = 8000  # 토큰 제한
        self.supported_extensions = ['.txt', '.md', '.json']
        
        print("📝 텍스트 처리기 초기화 완료")
    
    def process_text_file(self, file_path: Path) -> Document:
        """텍스트 파일을 Document 객체로 변환"""
        print(f"    📝 텍스트 파일 처리: {file_path.name}")
        
        try:
            extension = file_path.suffix.lower()
            
            if extension == '.json':
                content = self._process_json_file(file_path)
            elif extension == '.md':
                content = self._process_markdown_file(file_path)
            elif extension == '.txt':
                content = self._process_txt_file(file_path)
            else:
                # 기본 텍스트 처리
                content = self._process_txt_file(file_path)
            
            if not content or len(content.strip()) < 20:
                return self._create_empty_document(file_path, "내용이 너무 짧거나 비어있음")
            
            # 토큰 제한 적용
            if len(content) > self.max_content_length:
                content = content[:self.max_content_length] + "\n\n[내용이 길어 일부 생략됨]"
            
            print(f"    ✅ 텍스트 처리 완료: {len(content)}자")
            
            return Document(
                page_content=content,
                metadata={
                    "source": str(file_path),
                    "title": file_path.stem,
                    "file_type": extension[1:],  # 점 제거
                    "processed_at": datetime.now().isoformat(),
                    "category": self._infer_category_from_filename(file_path.name),
                    "content_length": len(content),
                    "original_length": len(content) if len(content) <= self.max_content_length else "truncated"
                }
            )
            
        except Exception as e:
            print(f"    ❌ 텍스트 처리 실패: {str(e)}")
            return self._create_error_document(file_path, str(e))
    
    def _process_txt_file(self, file_path: Path) -> str:
        """일반 텍스트 파일 처리"""
        try:
            # 여러 인코딩 시도
            encodings = ['utf-8', 'cp949', 'euc-kr', 'latin-1']
            
            for encoding in encodings:
                try:
                    with open(file_path, 'r', encoding=encoding) as f:
                        content = f.read()
                    
                    # 성공적으로 읽었으면 텍스트 정리 후 반환
                    return self._clean_text_content(content)
                    
                except UnicodeDecodeError:
                    continue
            
            # 모든 인코딩 실패
            print(f"    ⚠️ 인코딩 감지 실패: {file_path.name}")
            return f"파일 인코딩을 인식할 수 없습니다: {file_path.name}"
            
        except Exception as e:
            return f"텍스트 파일 읽기 실패: {str(e)}"
    
    def _process_markdown_file(self, file_path: Path) -> str:
        """마크다운 파일 처리"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 마크다운 특화 처리
            cleaned_content = self._clean_markdown_content(content)
            return cleaned_content
            
        except Exception as e:
            return f"마크다운 파일 처리 실패: {str(e)}"
    
    def _process_json_file(self, file_path: Path) -> str:
        """JSON 파일 처리"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # JSON 구조에 따른 텍스트 추출
            if isinstance(data, list):
                return self._extract_from_json_list(data)
            elif isinstance(data, dict):
                return self._extract_from_json_dict(data)
            else:
                return str(data)
                
        except json.JSONDecodeError as e:
            return f"JSON 파싱 오류: {str(e)}"
        except Exception as e:
            return f"JSON 파일 처리 실패: {str(e)}"
    
    def _clean_text_content(self, content: str) -> str:
        """일반 텍스트 정리"""
        if not content:
            return ""
        
        # 기본 정리
        content = content.strip()
        
        # 과도한 공백 정리
        content = re.sub(r'\n\s*\n\s*\n', '\n\n', content)  # 3개 이상 줄바꿈을 2개로
        content = re.sub(r' +', ' ', content)  # 여러 공백을 1개로
        content = re.sub(r'\t+', ' ', content)  # 탭을 공백으로
        
        # BOM 제거
        content = content.replace('\ufeff', '')
        
        return content
    
    def _clean_markdown_content(self, content: str) -> str:
        """마크다운 내용 정리"""
        if not content:
            return ""
        
        # 기본 텍스트 정리
        content = self._clean_text_content(content)
        
        # 마크다운 문법 간소화 (완전 제거하지 않고 읽기 쉽게)
        content = re.sub(r'^#{1,6}\s+', '■ ', content, flags=re.MULTILINE)  # 헤딩을 ■로
        content = re.sub(r'\*\*(.*?)\*\*', r'【\1】', content)  # 볼드를 【】로
        content = re.sub(r'\*(.*?)\*', r'〈\1〉', content)  # 이탤릭을 〈〉로
        content = re.sub(r'`(.*?)`', r'"\1"', content)  # 코드를 ""로
        
        # 링크 정리
        content = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', content)  # [텍스트](링크) → 텍스트
        
        # 리스트 마커 정리
        content = re.sub(r'^[\-\*\+]\s+', '• ', content, flags=re.MULTILINE)
        content = re.sub(r'^\d+\.\s+', '1. ', content, flags=re.MULTILINE)
        
        return content
    
    def _extract_from_json_list(self, data: List) -> str:
        """JSON 리스트에서 텍스트 추출"""
        content_parts = []
        
        for i, item in enumerate(data):
            if isinstance(item, dict):
                # 일반적인 텍스트 필드들 찾기
                text_fields = ['content', 'text', 'description', 'body', 'message', 'title']
                
                for field in text_fields:
                    if field in item and isinstance(item[field], str):
                        content_parts.append(f"[항목 {i+1}] {item[field]}")
                        break
                else:
                    # 텍스트 필드가 없으면 전체 JSON을 문자열로
                    content_parts.append(f"[항목 {i+1}] {json.dumps(item, ensure_ascii=False, indent=2)}")
            
            elif isinstance(item, str):
                content_parts.append(f"[항목 {i+1}] {item}")
            
            else:
                content_parts.append(f"[항목 {i+1}] {str(item)}")
        
        return "\n\n".join(content_parts)
    
    def _extract_from_json_dict(self, data: Dict) -> str:
        """JSON 딕셔너리에서 텍스트 추출"""
        # 의료 문서에서 자주 사용되는 필드명들
        priority_fields = [
            'content', 'text', 'description', 'body', 'message',
            'title', 'summary', 'abstract', 'procedure', 'symptoms',
            'treatment', 'diagnosis', 'medication', 'dosage'
        ]
        
        content_parts = []
        
        # 우선순위 필드 먼저 처리
        for field in priority_fields:
            if field in data and isinstance(data[field], str) and data[field].strip():
                content_parts.append(f"{field.upper()}: {data[field]}")
        
        # 나머지 필드들 처리
        for key, value in data.items():
            if key not in priority_fields:
                if isinstance(value, str) and value.strip():
                    content_parts.append(f"{key}: {value}")
                elif isinstance(value, (list, dict)):
                    content_parts.append(f"{key}: {json.dumps(value, ensure_ascii=False, indent=2)}")
                else:
                    content_parts.append(f"{key}: {str(value)}")
        
        return "\n\n".join(content_parts)
    
    def _infer_category_from_filename(self, filename: str) -> str:
        """파일명에서 카테고리 추론"""
        filename_lower = filename.lower()
        
        category_keywords = {
            "응급처치": ["응급", "emergency", "급성", "위급", "구급"],
            "약물정보": ["약물", "drug", "medication", "처방", "pharmacology"],
            "진단": ["진단", "diagnosis", "검사", "test", "examination"],
            "치료": ["치료", "treatment", "therapy", "수술", "surgery"],
            "간호": ["간호", "nursing", "care", "돌봄"],
            "내과": ["내과", "internal", "순환기", "호흡기", "소화기"],
            "외과": ["외과", "surgery", "수술", "정형외과", "신경외과"],
            "소아과": ["소아", "pediatric", "아동", "신생아"],
            "산부인과": ["산부인과", "obstetrics", "gynecology", "임신", "출산"],
            "가이드라인": ["guideline", "가이드", "지침", "프로토콜", "protocol"],
            "매뉴얼": ["manual", "매뉴얼", "안내서", "handbook"]
        }
        
        for category, keywords in category_keywords.items():
            if any(keyword in filename_lower for keyword in keywords):
                return category
        
        return "일반의학"
    
    def _create_empty_document(self, file_path: Path, reason: str) -> Document:
        """빈 문서 처리용 Document"""
        return Document(
            page_content=f"""
파일명: {file_path.name}
상태: 내용 없음

이유: {reason}

이 파일은 처리할 수 있는 텍스트 내용이 없습니다.
파일을 확인하고 내용을 추가한 후 다시 업로드하세요.
""",
            metadata={
                "source": str(file_path),
                "title": file_path.stem,
                "file_type": file_path.suffix[1:],
                "processed_at": datetime.now().isoformat(),
                "category": "빈파일",
                "status": "empty",
                "reason": reason
            }
        )
    
    def _create_error_document(self, file_path: Path, error: str) -> Document:
        """에러 처리용 Document"""
        return Document(
            page_content=f"""
파일명: {file_path.name}
상태: 처리 실패

오류: {error}

이 텍스트 파일을 처리하는 중 오류가 발생했습니다.
파일 형식이나 인코딩을 확인해보세요.

지원되는 형식: .txt, .md, .json
지원되는 인코딩: UTF-8, CP949, EUC-KR
""",
            metadata={
                "source": str(file_path),
                "title": file_path.stem,
                "file_type": file_path.suffix[1:],
                "processed_at": datetime.now().isoformat(),
                "category": "처리실패",
                "status": "error",
                "error": error
            }
        )
    
    def is_supported_file(self, file_path: Path) -> bool:
        """지원되는 파일 형식인지 확인"""
        return file_path.suffix.lower() in self.supported_extensions
    
    def get_stats(self) -> Dict:
        """텍스트 처리기 통계"""
        return {
            "processor_type": "TextProcessor",
            "supported_extensions": self.supported_extensions,
            "max_content_length": self.max_content_length
        }