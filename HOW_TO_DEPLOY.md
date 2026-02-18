# HMEAYC 系統 - GitHub 部署教學指南

這份指南將手把手教您如何將程式碼上傳到 GitHub，並透過 Streamlit Cloud 發布網站。

## 🛠️ 第一階段：準備檔案 (已完成)
確認您的資料夾中有以下關鍵檔案 (我已經幫您建立好了)：
1.  `app.py` (主程式)
2.  `requirements.txt` (告訴雲端要安裝 Python 套件)
3.  `packages.txt` (告訴雲端要安裝影片處理工具)
4.  `yolov8n-pose.pt` (AI 模型)

---

## ☁️ 第二階段：上傳到 GitHub (網頁版最簡單)

因為您可能不想用複雜的指令，我們直接用「網頁版」操作：

### 1. 建立新倉庫 (New Repository)
1.  登入 [GitHub](https://github.com/)。
2.  點擊右上角的 **+** 號，選擇 **New repository**。
3.  **Repository name** (倉庫名稱)：輸入 `hmeayc-app` (或您喜歡的名字)。
4.  **Public/Private**：選擇 **Public** (公開，免費版 Streamlit 只支援公開)。
5.  勾選 **Add a README file**。
6.  點擊最下方的 **Create repository**。

### 2. 上傳檔案 (Upload Files)
1.  進入剛建立好的倉庫頁面。
2.  點擊 **Add file** 按鈕 -> 選擇 **Upload files**。
3.  **拖曳檔案**：
    *   打開您電腦上的 `HMEAYC_Project` 資料夾。
    *   **全選** 所有檔案 (Ctrl+A)。
    *   **取消選取** (Ctrl+點擊) 以下檔案 (因為檔案太大或不需要)：
        *   `obs_video.mp4` (影片檔太大，GitHub 會拒收)
        *   `output_annotated...mp4`
        *   `temp_video.mp4`
        *   `__pycache__` 資料夾
        *   `.git` 資料夾 (如果有)
        *   `venv` 資料夾 (如果有)
    *   將剩下的檔案 **拖曳** 到 GitHub 網頁的方框中。
4.  等待檔案上傳進度條跑完。
5.  在下方 "Commit changes" (提交變更) 區塊：
    *   標題輸入：`Initial commit`
    *   點擊綠色的 **Commit changes** 按鈕。

---

## 🚀 第三階段：在 Streamlit Cloud 發布

### 1. 連結帳號
1.  前往 [Streamlit Cloud](https://streamlit.io/cloud)。
2.  點擊 **Sign Up** 或 **Log In**。
3.  選擇 **Continue with GitHub** (用 GitHub 帳號登入)。

### 2. 部署應用程式 (Deploy App)
1.  登入後，點擊右上角的 **New app**。
2.  **Repository**：選擇剛剛建立的 `hmeayc-app`。
3.  **Branch**：通常是 `main` (預設)。
4.  **Main file path**：輸入 `app.py`。
5.  點擊 **Deploy!**。

### 3. 等待安裝
*   系統會開始 "Cooking" (安裝環境)，這可能需要 2-3 分鐘。
*   您可以點擊右下角的 "Manage app" 查看安裝進度 (黑色終端機畫面)。
*   如果是第一次安裝，看到 `Running setup...` 是正常的。

---

## ❓ 常見問題 QA

**Q: 上傳時 GitHub 說 "File looks too big"？**
A: 您可能不小心拉到了 `.mp4` 影片檔。請取消上傳，確保只上傳程式碼 (.py)、設定檔 (.txt) 和模型 (.pt)。

**Q: 部署後網頁顯示 "ModuleNotFoundError"？**
A: 這通常是 `requirements.txt` 沒上傳成功，請檢查 GitHub 倉庫裡有沒有這個檔案。

**Q: 網頁跑出來了，但上傳影片後報錯？**
A: 雲端機器的資源有限 (只有 1GB RAM)，請上傳 **短一點** (例如 10-30秒) 的影片測試。
