# main.py (수정)
import os
from dotenv import load_dotenv
from rag_system import RAGSystem
from pathlib import Path

load_dotenv()

def main():
    print("🏥 Medical Chatbot 시작 (종료: 'quit' 입력)")
    
    try:
        print("🔄 시스템 초기화 중...")
        rag_system = RAGSystem()
        print("✅ RAG 시스템 준비 완료!")
        
        # 검색 소스 설정 (S3만 활성화, 나머지 비활성화)
        source_config = {
            "rag": False,     # 로컬 검색 비활성화
            "s3": True,       # S3 검색 활성화
            "medgemma": True, # MedGemma 활성화
            "pubmed": True    # PubMed 활성화
        }
        
        rag_system.configure_search_sources(source_config)
        
        # 시스템 상태 확인
        status = rag_system.get_system_status()
        print("\n🔍 검색 소스 설정:")
        for source, enabled in status["search_sources"].items():
            state = "✅ 활성화" if enabled else "❌ 비활성화"
            print(f"   - {source}: {state}")
        
        print("\n🎯 이제 의료 질문을 해보세요!")
        print("   예시: '당뇨병 관리 방법은?', '응급처치 절차는?'")
        
        user_id = "console_user"
        
        while True:
            question = input("\n❓ 질문: ").strip()
            
            # 종료 명령
            if question.lower() in ['quit', 'exit', '종료', 'q']:
                print("👋 Medical Chatbot을 종료합니다.")
                break
            
            # 특수 명령: 검색 소스 설정
            if question.startswith("/config"):
                cmd_parts = question.split()
                if len(cmd_parts) >= 3:
                    src = cmd_parts[1]
                    enabled = cmd_parts[2].lower() in ['on', 'true', '1']
                    
                    config = {src: enabled}
                    new_config = rag_system.configure_search_sources(config)
                    
                    print(f"\n🔧 검색 소스 설정 변경:")
                    for s, e in new_config.items():
                        state = "✅ 활성화" if e else "❌ 비활성화"
                        print(f"   - {s}: {state}")
                continue
            
            # 특수 명령: 시스템 상태 확인
            if question.lower() in ['/status', '/stat', '/stats']:
                status = rag_system.get_system_status()
                
                print("\n📊 시스템 상태:")
                print("\n🔍 검색 소스:")
                for source, enabled in status["search_sources"].items():
                    state = "✅ 활성화" if enabled else "❌ 비활성화"
                    print(f"   - {source}: {state}")
                
                if status["s3_stats"]:
                    print("\n📦 S3 검색 통계:")
                    s3_stats = status["s3_stats"]
                    print(f"   - 총 검색: {s3_stats['total_searches']}회")
                    print(f"   - 성공률: {s3_stats['success_rate']}")
                    print(f"   - 평균 응답시간: {s3_stats['average_response_time']}")
                
                continue
                
            if not question:
                continue
            
            print("🤔 답변 생성 중...")
            
            try:
                result = rag_system.run_graph(question, user_id)
                
                # 답변 출력
                if isinstance(result, dict) and "answer" in result:
                    print(f"\n🏥 답변:\n{result['answer']}")
                else:
                    print(f"\n🏥 답변:\n{result}")
                
            except Exception as e:
                print(f"❌ 답변 생성 실패: {e}")
                print("💡 시스템 문제가 있을 수 있습니다")
    
    except KeyboardInterrupt:
        print("\n👋 사용자가 종료했습니다.")
    except Exception as e:
        print(f"❌ 에러 발생: {e}")
        print("💡 OPENAI_API_KEY가 설정되었는지 확인하세요")

if __name__ == "__main__":
    main()