@echo off
echo Starting Basic Lithography Hotspot Detection App...
echo.
cd /d "C:\Users\Abhinav\Desktop\Mainprojects"
call ".venv\Scripts\activate.bat"
echo Virtual environment activated
echo.
echo Starting Basic Streamlit App on http://localhost:8501
echo Press Ctrl+C to stop the server
echo.
python -m streamlit run app_working.py --server.port=8501
pause
