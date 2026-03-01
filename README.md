---
title: Hmeayc App
emoji: 🩰
colorFrom: pink
colorTo: indigo
sdk: docker
app_file: app.py
pinned: false
---

# HMEAYC - AI 幼兒肢體動作分析系統

這是一個基於 AI (YOLOv8 + MediaPipe) 的幼兒肢體動作分析系統，專為幼兒園教師設計，能夠自動分析課堂影片中的幼兒動作、互動與專注力。

## 🌟 功能特色

* **AI 動作評分**：自動計算動作活躍度 (1-5分)。
* **群體同步率**：分析全班動作的一致性與方向性。
* **專注力分析**：透過頭部轉向判斷是否專注於老師。
* **社交圖譜**：繪製幼兒間的互動網絡，找出核心人物。
* **AI 評語**：根據數據自動生成個別化的觀察評語。

## 🚀 如何部署 (Streamlit Cloud)

此專案已針對 Streamlit Cloud 最佳化。

1. **上傳至 GitHub**：
    * 將此資料夾所有檔案 (除了 .gitgnore 排除的影片檔) 上傳至您的 GitHub Repository。
2. **設定 Streamlit Cloud**：
    * 登入 [Streamlit Cloud](https://streamlit.io/cloud)。
    * 點選 "New app"。
    * 選擇您的 Repository。
    * **Main file path** 輸入 `app.py`。
    * 點選 **Deploy**！

## 📦 依賴套件

系統會自動讀取以下設定檔進行安裝：

* `requirements.txt`: Python 套件 (Streamlit, YOLO, OpenCV-Headless)
* `packages.txt`: 系統套件 (FFmpeg, LibGL)

## 📁 檔案結構

* `app.py`: 主程式可以直接執行。
* `yolov8n-pose.pt`: AI 模型權重。
* `botsort_custom.yaml`: 追蹤演算法設定。

---
*Created for HMEAYC Project*
