#!/bin/bash

echo "✅ React 서버 실행 중..."
cd frontend
npm run dev &
REACT_PID=$!
cd ..

echo "✅ Streamlit 서버 실행 중..."
streamlit run streamlit_app.py --server.port=8514 --server.enableCORS false --server.enableXsrfProtection false &
STREAMLIT_PID=$!

echo "🟢 모든 서버 실행 완료!"
echo "🌐 React:     http://localhost:5173"
echo "🌐 Streamlit: http://localhost:8514"

echo ""
echo "중지하려면 Ctrl+C 를 누르세요."
wait $REACT_PID $STREAMLIT_PID
