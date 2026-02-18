@echo off
cd /d %~dp0
echo Installing dependencies...
pip install -r requirements.txt
echo Starting App..."
echo 正在準備啟動 HMEAYC 專業觀察系統...
echo 正在嘗試自動開啟瀏覽器...

REM 嘗試先打開瀏覽器
start "" "http://localhost:8501"

echo 正在啟動伺服器，請稍候...
echo 如果瀏覽器未自動連線，請重新整理頁面。

REM 執行 Streamlit
python -m streamlit run app.py --server.headless false

if %errorlevel% neq 0 (
    echo ========================================================
    echo 啟動失敗！可能是環境變數未設定或未安裝 Streamlit。
    echo 請嘗試手動開啟 cmd 並輸入: python -m streamlit run app.py
    echo ========================================================
    pause
)
pause
