# main.py (수정)
import os
from dotenv import load_dotenv
from rag_system import RAGSystem
from pathlib import Path
import traceback

load_dotenv()

def main():
    print("🏥 Medical Chatbot 시작 (종료: 'quit' 입력)")
    
    try:
        print("🔄 시스템 초기화 중...")
        rag_system = RAGSystem()
        print("✅ RAG 시스템 준비 완료!")
        
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
                error_trace = traceback.format_exc()
                print(f"❌ 답변 생성 실패: {str(e)}\n\n상세 오류:\n{error_trace}")
                print("💡 시스템 문제가 있을 수 있습니다")
                
        print(result["language_info"])
    
    except KeyboardInterrupt:
        print("\n👋 사용자가 종료했습니다.")
    except Exception as e:
        print(f"❌ 에러 발생: {e}")
        print("상세 에러 정보:")
        traceback.print_exc()

if __name__ == "__main__":
    main()
