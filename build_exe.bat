
@echo off
echo Installing PyInstaller...
pip install pyinstaller

echo Cleaning up previous builds...
rmdir /s /q build dist
del /q *.spec

echo Building Executable...
echo This may take a while (5-10 minutes)...

pyinstaller --noconfirm --onefile --windowed ^
    --name "HMEAYC_Observer" ^
    --hidden-import=streamlit ^
    --hidden-import=altair ^
    --hidden-import=pandas ^
    --hidden-import=numpy ^
    --hidden-import=cv2 ^
    --hidden-import=ultralytics ^
    --collect-all=streamlit ^
    --collect-all=ultralytics ^
    --collect-all=altair ^
    --copy-metadata=streamlit ^
    --copy-metadata=ultralytics ^
    --copy-metadata=tqdm ^
    --copy-metadata=regex ^
    --copy-metadata=requests ^
    --copy-metadata=packaging ^
    --copy-metadata=filelock ^
    --copy-metadata=numpy ^
    --add-data "app.py;." ^
    --add-data "yolov8n-pose.pt;." ^
    --add-data "botsort_custom.yaml;." ^
    run_app.py

echo Build Complete!
pause
