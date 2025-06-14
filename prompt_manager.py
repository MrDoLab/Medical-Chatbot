# prompt_manager.py
import json
import os
from typing import Dict, List, Any, Optional
from pathlib import Path
from datetime import datetime

class PromptManager:
    """프롬프트 관리 클래스 - 저장, 로드, 프리셋 관리"""
    
    def __init__(self, preset_dir: str = "./prompt_presets"):
        """
        프롬프트 관리자 초기화
        
        Args:
            preset_dir: 프리셋 저장 디렉토리
        """
        self.preset_dir = Path(preset_dir)
        self.preset_dir.mkdir(exist_ok=True)
        
        # 기본 프롬프트 타입
        self.prompt_types = [
            "RAG_SYSTEM_PROMPT", 
            "ROUTER_SYSTEM_PROMPT",
            "GRADER_SYSTEM_PROMPT",
            "HALLUCINATION_SYSTEM_PROMPT",
            "REWRITER_SYSTEM_PROMPT"
        ]
        
        # 프리셋 목록 캐시
        self._preset_list_cache = None
        
    def get_prompt(self, prompt_type: str, config_obj) -> str:
        """
        현재 설정에서 프롬프트 가져오기
        
        Args:
            prompt_type: 프롬프트 타입 (예: "RAG_SYSTEM_PROMPT")
            config_obj: Config 클래스 객체
            
        Returns:
            프롬프트 텍스트
        """
        if hasattr(config_obj, prompt_type):
            return getattr(config_obj, prompt_type)
        return f"프롬프트 '{prompt_type}'를 찾을 수 없습니다."
    
    def update_prompt(self, prompt_type: str, new_content: str, config_obj) -> bool:
        """
        프롬프트 업데이트
        
        Args:
            prompt_type: 프롬프트 타입
            new_content: 새 프롬프트 내용
            config_obj: Config 클래스 객체
            
        Returns:
            성공 여부
        """
        try:
            if hasattr(config_obj, prompt_type):
                setattr(config_obj, prompt_type, new_content)
                return True
            return False
        except Exception as e:
            print(f"프롬프트 업데이트 실패: {str(e)}")
            return False
    
    def save_preset(self, preset_name: str, prompts: Dict[str, str]) -> bool:
        """
        프롬프트 프리셋 저장
        
        Args:
            preset_name: 프리셋 이름
            prompts: 프롬프트 타입별 내용
            
        Returns:
            성공 여부
        """
        try:
            # 파일명 안전하게 처리
            safe_name = preset_name.replace(" ", "_").lower()
            preset_file = self.preset_dir / f"{safe_name}.json"
            
            preset_data = {
                "name": preset_name,
                "created_at": datetime.now().isoformat(),
                "prompts": prompts
            }
            
            with open(preset_file, 'w', encoding='utf-8') as f:
                json.dump(preset_data, f, ensure_ascii=False, indent=2)
            
            # 캐시 무효화
            self._preset_list_cache = None
            
            return True
        except Exception as e:
            print(f"프리셋 저장 실패: {str(e)}")
            return False
    
    def load_preset(self, preset_name: str) -> Optional[Dict[str, str]]:
        """
        프롬프트 프리셋 로드
        
        Args:
            preset_name: 프리셋 이름
            
        Returns:
            프롬프트 딕셔너리 또는 None
        """
        try:
            # 파일명 안전하게 처리
            safe_name = preset_name.replace(" ", "_").lower()
            preset_file = self.preset_dir / f"{safe_name}.json"
            
            if not preset_file.exists():
                return None
            
            with open(preset_file, 'r', encoding='utf-8') as f:
                preset_data = json.load(f)
            
            return preset_data.get("prompts", {})
        except Exception as e:
            print(f"프리셋 로드 실패: {str(e)}")
            return None
    
    def get_preset_list(self) -> List[Dict[str, Any]]:
        """
        사용 가능한 프리셋 목록 가져오기
        
        Returns:
            프리셋 정보 목록
        """
        # 캐시된 목록 사용
        if self._preset_list_cache is not None:
            return self._preset_list_cache
        
        presets = []
        try:
            for preset_file in self.preset_dir.glob("*.json"):
                try:
                    with open(preset_file, 'r', encoding='utf-8') as f:
                        preset_data = json.load(f)
                    
                    presets.append({
                        "name": preset_data.get("name", preset_file.stem),
                        "created_at": preset_data.get("created_at", "Unknown"),
                        "filename": preset_file.name,
                        "prompt_count": len(preset_data.get("prompts", {}))
                    })
                except:
                    # 잘못된 형식의 파일은 무시
                    continue
            
            # 최신순 정렬
            presets.sort(key=lambda x: x.get("created_at", ""), reverse=True)
            
            # 캐시 업데이트
            self._preset_list_cache = presets
            
            return presets
        except Exception as e:
            print(f"프리셋 목록 로드 실패: {str(e)}")
            return []
    
    def delete_preset(self, preset_name: str) -> bool:
        """
        프리셋 삭제
        
        Args:
            preset_name: 프리셋 이름
            
        Returns:
            성공 여부
        """
        try:
            # 파일명 안전하게 처리
            safe_name = preset_name.replace(" ", "_").lower()
            preset_file = self.preset_dir / f"{safe_name}.json"
            
            if preset_file.exists():
                preset_file.unlink()
                # 캐시 무효화
                self._preset_list_cache = None
                return True
            return False
        except Exception as e:
            print(f"프리셋 삭제 실패: {str(e)}")
            return False