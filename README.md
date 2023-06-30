# 人臉辨識視窗程式

這個程式是一個基於人臉辨識的視窗應用程式，用於進行人臉的檢測和影像處理。它提供了多種功能，包括人臉檢測、特徵點檢測、影像處理等。

## 功能特色

- 攝影機即時讀取視訊並顯示在視窗中。
- 拍攝照片並保存為圖片檔案。
- 影像處理功能：翻轉、對比度增強、模糊化、雜訊去除、侵蝕、膨脹、灰階、邊緣檢測、二值化。
- 人臉追蹤功能：檢測和追蹤視訊中的人臉。
- 人臉特徵點檢測：檢測人臉區域並顯示特徵點。

## 使用方式

1. 安裝所需的函式庫和模型檔案。
2. 開啟程式，它會建立一個視窗，顯示攝影機的影像。
3. 使用按鈕執行相應的功能，例如拍攝照片、翻轉影像、增強人臉對比度等。
4. 可以根據需要在程式中設定功能的開關。
5. 程式會持續從攝影機中讀取影像，並在視窗中更新顯示。

請注意，執行此程式需要安裝相關的函式庫（OpenCV、dlib、PIL等）和準備人臉辨識模型檔案（如haarcascade_frontalface_default.xml和shape_predictor_68_face_landmarks.dat）。

## 執行環境要求

- 支援攝影機的電腦或設備。
- 安裝Python 3.x。
- 安裝所需的函式庫（詳見程式中的相依套件）。
