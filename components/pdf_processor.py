# components/pdf_processor.py
"""
PDF 전용 처리기 - 텍스트 추출 + OCR 하이브리드 처리
"""

import fitz  # PyMuPDF
import pandas as pd
import pytesseract
from PIL import Image
import io
import re
from typing import List, Dict, Optional
from pathlib import Path
from langchain_core.documents import Document
from datetime import datetime

class PDFProcessor:
    """PDF 문서 전용 처리기 - 텍스트/OCR 하이브리드"""
    
    def __init__(self):
        """PDF 처리기 초기화"""
        self.max_text_pages = 100    # 텍스트 추출 최대 페이지
        self.max_ocr_pages = 100     # OCR 처리 최대 페이지
        self.text_threshold = 200     # 텍스트 추출 성공 임계값 (글자수)
        self.ocr_threshold = 30       # OCR 페이지별 최소 글자수
        
        print("📄 PDF 처리기 초기화 완료")
    
    def process_pdf(self, file_path: Path) -> Document:
        """PDF 파일을 Document 객체로 변환 (하이브리드 처리)"""
        print(f"    🔍 PDF 분석 시작: {file_path.name}")
        
        try:
            doc = fitz.open(str(file_path))
            
            # 1단계: 텍스트 추출 시도
            text_result = self._try_text_extraction(doc)
            
            if text_result["success"]:
                doc.close()
                print(f"    ✅ 텍스트 추출 성공: {len(text_result['content'])}자")
                return self._create_document(file_path, text_result["content"], "text_extraction")
            
            # 2단계: OCR 처리 시도
            print(f"    📷 스캔된 PDF 감지 - OCR 시도 중...")
            ocr_result = self._try_ocr_extraction(doc)
            
            doc.close()
            
            if ocr_result["success"]:
                print(f"    ✅ OCR 추출 성공: {len(ocr_result['content'])}자")
                return self._create_document(file_path, ocr_result["content"], "ocr_extraction")
            else:
                print(f"    ❌ OCR 실패: {ocr_result['error']}")
                return self._create_fallback_document(file_path, ocr_result["error"])
                
        except Exception as e:
            print(f"    ❌ PDF 처리 실패: {str(e)}")
            return self._create_fallback_document(file_path, str(e))
    
    def _try_text_extraction(self, doc) -> Dict:
        """1단계: 텍스트 + 표 추출 시도"""
        try:
            extracted_content = []
            total_text_length = 0
            
            for page_num in range(min(self.max_text_pages, len(doc))):
                page = doc[page_num]
                
                # 텍스트 추출
                text = page.get_text()
                
                # 표 추출
                tables = self._extract_tables_from_page(page)
                
                # 페이지 내용 구성
                page_content = []
                
                if text.strip():
                    cleaned_text = self._clean_pdf_text(text)
                    if len(cleaned_text) > 50:  # 의미있는 텍스트
                        page_content.append(f"=== 페이지 {page_num + 1} ===")
                        page_content.append(cleaned_text)
                        total_text_length += len(cleaned_text)
                
                if tables:
                    page_content.append("\n[표 데이터]")
                    for i, table in enumerate(tables):
                        page_content.append(f"표 {i+1}:")
                        page_content.append(table)
                        total_text_length += len(table)
                
                if page_content:
                    extracted_content.append("\n".join(page_content))
            
            # 텍스트 추출 성공 기준
            if total_text_length >= self.text_threshold:
                full_content = "\n\n".join(extracted_content)
                
                # 토큰 제한
                if len(full_content) > 8000:
                    full_content = full_content[:8000] + "\n\n[내용이 길어 일부 생략됨]"
                
                return {
                    "success": True,
                    "content": full_content,
                    "method": "text_extraction",
                    "pages_processed": len(extracted_content),
                    "total_chars": total_text_length
                }
            else:
                return {
                    "success": False,
                    "reason": f"텍스트 부족 ({total_text_length}자 < {self.text_threshold}자)"
                }
                
        except Exception as e:
            return {
                "success": False,
                "reason": f"텍스트 추출 오류: {str(e)}"
            }
    
    def _try_ocr_extraction(self, doc) -> Dict:
        """2단계: OCR 추출 시도"""
        try:
            # OCR 의존성 체크
            if not self._check_ocr_dependencies():
                return {
                    "success": False,
                    "error": "OCR 의존성 미설치 (pytesseract, Pillow)"
                }
            
            extracted_pages = []
            max_pages = min(self.max_ocr_pages, len(doc))
            
            print(f"      🔄 OCR 처리중... ({max_pages}페이지)")
            
            for page_num in range(max_pages):
                page = doc[page_num]
                
                # 페이지를 고해상도 이미지로 변환
                matrix = fitz.Matrix(2.0, 2.0)  # 2배 확대
                pix = page.get_pixmap(matrix=matrix)
                img_data = pix.tobytes("png")
                
                # PIL Image로 변환
                image = Image.open(io.BytesIO(img_data))
                
                # 이미지 전처리
                image = self._preprocess_image_for_ocr(image)
                
                # Tesseract OCR 실행
                ocr_text = pytesseract.image_to_string(
                    image,
                    lang='kor+eng',  # 한글+영어
                    config='--psm 6 -c preserve_interword_spaces=1'
                )
                
                # 정리 및 검증
                cleaned_ocr = self._clean_ocr_text(ocr_text)
                
                if len(cleaned_ocr) > self.ocr_threshold:
                    extracted_pages.append(f"=== 페이지 {page_num + 1} (OCR) ===\n{cleaned_ocr}")
                
                # 메모리 정리
                image.close()
                pix = None
                
                print(f"        페이지 {page_num + 1}: {len(cleaned_ocr)}자 추출")
            
            if extracted_pages:
                content = "\n\n".join(extracted_pages)
                
                # 토큰 제한
                if len(content) > 8000:
                    content = content[:8000] + "\n\n[OCR 내용이 길어 일부 생략됨]"
                
                return {
                    "success": True,
                    "content": content,
                    "method": "ocr_extraction",
                    "pages_processed": len(extracted_pages)
                }
            else:
                return {
                    "success": False,
                    "error": "OCR로 추출된 의미있는 텍스트 없음"
                }
                
        except Exception as e:
            return {
                "success": False,
                "error": f"OCR 처리 오류: {str(e)}"
            }
    
    def _extract_tables_from_page(self, page) -> List[str]:
        """페이지에서 표 데이터 추출"""
        tables = []
        
        try:
            table_list = page.find_tables()
            
            for table in table_list:
                table_data = table.extract()
                
                if table_data and len(table_data) > 1:  # 헤더 + 최소 1행
                    df = pd.DataFrame(table_data[1:], columns=table_data[0])
                    df = df.fillna("")
                    table_text = self._format_table_as_text(df)
                    tables.append(table_text)
            
        except Exception as e:
            print(f"      ⚠️ 표 추출 실패: {str(e)}")
        
        return tables
    
    def _format_table_as_text(self, df: pd.DataFrame) -> str:
        """DataFrame을 텍스트로 변환"""
        if df.empty:
            return ""
        
        try:
            if len(df) > 20:
                summary_df = df.head(20)
                table_text = summary_df.to_string(index=False, max_cols=6)
                table_text += f"\n... (총 {len(df)}행 중 20행만 표시)"
            else:
                table_text = df.to_string(index=False, max_cols=8)
            
            return table_text
            
        except Exception:
            rows = []
            for _, row in df.iterrows():
                row_text = " | ".join([str(cell)[:50] for cell in row.values])
                rows.append(row_text)
            
            return "\n".join(rows[:10])
    
    def _clean_pdf_text(self, text: str) -> str:
        """PDF 텍스트 정리"""
        # 불필요한 공백 제거
        text = re.sub(r'\n\s*\n', '\n\n', text)
        text = re.sub(r' +', ' ', text)
        
        # 페이지 번호 제거
        text = re.sub(r'\n-?\s*\d+\s*-?\n', '\n', text)
        text = re.sub(r'\nPage\s+\d+\n', '\n', text)
        
        # 의료 문서 특화 패턴 제거
        text = re.sub(r'\n의료.*?가이드라인\n', '\n', text)
        text = re.sub(r'\n\d{4}년\s*\d{1,2}월\n', '\n', text)
        
        return text.strip()
    
    def _check_ocr_dependencies(self) -> bool:
        """OCR 의존성 확인"""
        try:
            # Tesseract 실행 파일 확인
            pytesseract.get_tesseract_version()
            return True
            
        except pytesseract.TesseractNotFoundError:
            print("      ❌ Tesseract 실행파일 미설치")
            print("         Windows: https://github.com/UB-Mannheim/tesseract/wiki")
            print("         macOS: brew install tesseract")
            print("         Ubuntu: apt install tesseract-ocr tesseract-ocr-kor")
            return False
        except Exception as e:
            print(f"      ❌ OCR 환경 확인 실패: {str(e)}")
            return False
    
    def _preprocess_image_for_ocr(self, image: Image.Image) -> Image.Image:
        """OCR을 위한 이미지 전처리"""
        try:
            # 그레이스케일 변환
            if image.mode != 'L':
                image = image.convert('L')
            
            # 크기 조정 (너무 작으면 확대)
            width, height = image.size
            if width < 1000:
                scale_factor = 1000 / width
                new_width = int(width * scale_factor)
                new_height = int(height * scale_factor)
                image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
            
            return image
            
        except Exception:
            return image
    
    def _clean_ocr_text(self, ocr_text: str) -> str:
        """OCR 결과 텍스트 정리"""
        if not ocr_text:
            return ""
        
        text = ocr_text.strip()
        
        # OCR 특유의 노이즈 제거
        text = re.sub(r'[|＼/]{3,}', '', text)  # 선 패턴 제거
        text = re.sub(r'[-_=]{5,}', '', text)   # 긴 구분선 제거
        text = re.sub(r'\s+', ' ', text)        # 여러 공백을 하나로
        text = re.sub(r'\n\s*\n', '\n\n', text) # 여러 줄바꿈을 두개로
        
        # 의미없는 단일 문자 줄 제거
        lines = text.split('\n')
        cleaned_lines = []
        
        for line in lines:
            line = line.strip()
            if len(line) > 2 or line.isdigit():
                cleaned_lines.append(line)
        
        return '\n'.join(cleaned_lines)
    
    def _create_document(self, file_path: Path, content: str, method: str) -> Document:
        """Document 객체 생성"""
        return Document(
            page_content=content,
            metadata={
                "source": str(file_path),
                "title": file_path.stem,
                "file_type": "pdf",
                "extraction_method": method,
                "processed_at": datetime.now().isoformat(),
                "category": self._infer_category_from_filename(file_path.name),
                "content_length": len(content)
            }
        )
    
    def _create_fallback_document(self, file_path: Path, error: str) -> Document:
        """처리 실패시 폴백 Document"""
        fallback_content = f"""
파일명: {file_path.name}
상태: PDF 처리 실패

오류 내용: {error}

이 PDF 문서는 텍스트 추출과 OCR 모두 실패했습니다.

가능한 원인:
1. 이미지 품질이 너무 낮음
2. 특수한 PDF 형식 또는 암호화
3. OCR 환경 설정 문제
4. 손상된 PDF 파일

권장 조치:
- 문서를 직접 확인: {file_path}
- 텍스트 형태로 재작성 후 업로드
- PDF 품질 개선 후 재업로드

⚠️ 중요한 의료 정보가 포함되어 있을 수 있으니 수동 확인 필요
"""
        
        return Document(
            page_content=fallback_content,
            metadata={
                "source": str(file_path),
                "title": file_path.stem,
                "file_type": "pdf",
                "extraction_method": "failed",
                "error": error,
                "processed_at": datetime.now().isoformat(),
                "category": "처리실패",
                "requires_manual_review": True
            }
        )
    
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
            "산부인과": ["산부인과", "obstetrics", "gynecology", "임신", "출산"]
        }
        
        for category, keywords in category_keywords.items():
            if any(keyword in filename_lower for keyword in keywords):
                return category
        
        return "일반의학"
    
    def get_stats(self) -> Dict:
        """PDF 처리기 통계"""
        return {
            "processor_type": "PDFProcessor",
            "max_text_pages": self.max_text_pages,
            "max_ocr_pages": self.max_ocr_pages,
            "text_threshold": self.text_threshold,
            "ocr_threshold": self.ocr_threshold
        }