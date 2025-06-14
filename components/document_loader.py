# components/document_loader.py
"""
문서 로딩 총괄 관리자 - 파일 타입별 적절한 프로세서에게 위임
"""

from typing import List, Dict, Any
from pathlib import Path
from langchain_core.documents import Document
from datetime import datetime

from components.pdf_processor import PDFProcessor
from components.text_processor import TextProcessor

class DocumentLoader:
    """문서 로딩 총괄 관리자"""
    
    def __init__(self):
        """문서 로더 초기화"""
        self.pdf_processor = PDFProcessor()
        self.text_processor = TextProcessor()
        
        # 지원 파일 형식 정의
        self.pdf_extensions = ['.pdf']
        self.text_extensions = ['.txt', '.md', '.json']
        
        # 통계
        self.stats = {
            "total_files_processed": 0,
            "successful_loads": 0,
            "failed_loads": 0,
            "pdf_files": 0,
            "text_files": 0,
            "skipped_files": 0
        }
        
        print("📚 문서 로더 초기화 완료")
        print(f"   📄 PDF 처리기: 준비됨")
        print(f"   📝 텍스트 처리기: 준비됨")
    
    def load_documents_from_directory(self, directory_path: str) -> List[Document]:
        """디렉토리에서 모든 문서 로드"""
        directory = Path(directory_path)
        
        if not directory.exists():
            print(f"❌ 디렉토리가 존재하지 않습니다: {directory_path}")
            return []
        
        print(f"📁 문서 로딩 시작: {directory_path}")
        
        # 처리할 파일들 수집
        all_files = []
        for extension in self.pdf_extensions + self.text_extensions:
            all_files.extend(directory.rglob(f"*{extension}"))
        
        if not all_files:
            print(f"📭 처리할 파일이 없습니다 (지원 형식: {self.pdf_extensions + self.text_extensions})")
            return []
        
        print(f"📋 처리 대상: {len(all_files)}개 파일")
        
        # 파일별 처리
        loaded_documents = []
        
        for file_path in all_files:
            try:
                document = self.load_single_file(file_path)
                
                if document:
                    loaded_documents.append(document)
                    self.stats["successful_loads"] += 1
                    
                    # 파일 타입별 카운트
                    if file_path.suffix.lower() in self.pdf_extensions:
                        self.stats["pdf_files"] += 1
                    elif file_path.suffix.lower() in self.text_extensions:
                        self.stats["text_files"] += 1
                else:
                    self.stats["failed_loads"] += 1
                
                self.stats["total_files_processed"] += 1
                
            except Exception as e:
                print(f"    ❌ 처리 실패: {file_path.name} - {str(e)}")
                self.stats["failed_loads"] += 1
                self.stats["total_files_processed"] += 1
        
        # 결과 요약
        self._print_loading_summary(loaded_documents)
        
        return loaded_documents
    
    def load_single_file(self, file_path: Path) -> Document:
        """단일 파일 로드"""
        if not file_path.exists():
            print(f"    ❌ 파일이 존재하지 않음: {file_path}")
            return None
        
        extension = file_path.suffix.lower()
        
        # 파일 크기 체크 (100MB 제한)
        file_size = file_path.stat().st_size
        if file_size > 100 * 1024 * 1024:  # 100MB
            print(f"    ⚠️ 파일이 너무 큼: {file_path.name} ({file_size / (1024*1024):.1f}MB)")
            return self._create_oversized_document(file_path, file_size)
        
        # 파일 타입별 처리
        if extension in self.pdf_extensions:
            return self.pdf_processor.process_pdf(file_path)
        
        elif extension in self.text_extensions:
            return self.text_processor.process_text_file(file_path)
        
        else:
            print(f"    ⚠️ 지원되지 않는 파일 형식: {file_path.name} ({extension})")
            self.stats["skipped_files"] += 1
            return self._create_unsupported_document(file_path, extension)
    
    def _create_oversized_document(self, file_path: Path, file_size: int) -> Document:
        """크기 초과 파일용 Document"""
        size_mb = file_size / (1024 * 1024)
        
        return Document(
            page_content=f"""
파일명: {file_path.name}
상태: 파일 크기 초과

파일 크기: {size_mb:.1f}MB (제한: 100MB)

이 파일은 크기가 너무 커서 처리할 수 없습니다.

권장 조치:
1. 파일을 여러 개로 분할
2. 압축률을 높여 크기 줄이기
3. 핵심 내용만 별도 파일로 추출

원본 파일: {file_path}
""",
            metadata={
                "source": str(file_path),
                "title": file_path.stem,
                "file_type": file_path.suffix[1:],
                "processed_at": datetime.now().isoformat(),
                "category": "크기초과",
                "status": "oversized",
                "file_size_mb": round(size_mb, 1),
                "requires_manual_processing": True
            }
        )
    
    def _create_unsupported_document(self, file_path: Path, extension: str) -> Document:
        """지원되지 않는 파일용 Document"""
        return Document(
            page_content=f"""
파일명: {file_path.name}
상태: 지원되지 않는 형식

파일 형식: {extension}

현재 지원되는 파일 형식:
• PDF: {', '.join(self.pdf_extensions)}
• 텍스트: {', '.join(self.text_extensions)}

권장 조치:
1. 지원되는 형식으로 변환
2. 텍스트 내용을 .txt 파일로 추출
3. 내용을 복사해서 새 파일 생성

원본 파일: {file_path}
""",
            metadata={
                "source": str(file_path),
                "title": file_path.stem,
                "file_type": extension[1:] if extension else "unknown",
                "processed_at": datetime.now().isoformat(),
                "category": "미지원형식",
                "status": "unsupported",
                "requires_conversion": True
            }
        )
    
    def _print_loading_summary(self, documents: List[Document]):
        """로딩 결과 요약 출력"""
        print(f"\n📊 문서 로딩 완료:")
        print(f"   📄 총 처리: {self.stats['total_files_processed']}개 파일")
        print(f"   ✅ 성공: {self.stats['successful_loads']}개")
        print(f"   ❌ 실패: {self.stats['failed_loads']}개")
        print(f"   ⏭️ 건너뜀: {self.stats['skipped_files']}개")
        
        if self.stats['successful_loads'] > 0:
            print(f"\n📋 파일 타입별:")
            print(f"   📄 PDF: {self.stats['pdf_files']}개")
            print(f"   📝 텍스트: {self.stats['text_files']}개")
        
        # 카테고리별 통계
        if documents:
            categories = {}
            total_content_length = 0
            
            for doc in documents:
                category = doc.metadata.get("category", "미분류")
                categories[category] = categories.get(category, 0) + 1
                
                content_length = doc.metadata.get("content_length", 0)
                if isinstance(content_length, int):
                    total_content_length += content_length
            
            print(f"\n🏷️ 카테고리별:")
            for category, count in sorted(categories.items()):
                print(f"   • {category}: {count}개")
            
            print(f"\n📊 총 텍스트 길이: {total_content_length:,}자")
    
    def get_supported_extensions(self) -> List[str]:
        """지원되는 파일 확장자 목록"""
        return self.pdf_extensions + self.text_extensions
    
    def is_supported_file(self, file_path: Path) -> bool:
        """파일이 지원되는 형식인지 확인"""
        extension = file_path.suffix.lower()
        return extension in self.get_supported_extensions()
    
    def get_stats(self) -> Dict[str, Any]:
        """문서 로더 통계"""
        return {
            "loader_type": "DocumentLoader",
            "processing_stats": self.stats.copy(),
            "supported_extensions": {
                "pdf": self.pdf_extensions,
                "text": self.text_extensions
            },
            "processors": {
                "pdf_processor": self.pdf_processor.get_stats(),
                "text_processor": self.text_processor.get_stats()
            }
        }
    
    def reset_stats(self):
        """통계 초기화"""
        self.stats = {
            "total_files_processed": 0,
            "successful_loads": 0,
            "failed_loads": 0,
            "pdf_files": 0,
            "text_files": 0,
            "skipped_files": 0
        }
        print("📊 통계가 초기화되었습니다")