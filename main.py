# main.py
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
                
        # 의료 문서 로드 체크
        medical_docs_path = "./medical_docs"
        if Path(medical_docs_path).exists():
            print(f"📚 의료 문서 로딩 중: {medical_docs_path}")
            count = rag_system.load_medical_documents(medical_docs_path)
            print(f"✅ {count}개 의료 문서 로드 완료")
        else:
            print("⚠️ 의료 문서 폴더가 없습니다. 웹 검색 위주로 작동합니다.")
        
        print("\n🎯 이제 의료 질문을 해보세요!")
        print("   예시: '당뇨병 관리 방법은?', '응급처치 절차는?'")
        
        user_id = "console_user"
        
        while True:
            question = input("\n❓ 질문: ").strip()
            
            if question.lower() in ['quit', 'exit', '종료', 'q']:
                print("👋 Medical Chatbot을 종료합니다.")
                break
            
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