import cv2
import dlib
import tkinter as tk
from PIL import Image, ImageTk
import numpy as np

window = tk.Tk()
window.title("人臉辨識視窗")
window.geometry("1024x768")

cap = cv2.VideoCapture(0)

label = tk.Label(window)
label.pack(pady=10)

button_frame = tk.Frame(window)
button_frame.pack(pady=10)

btn_capture = tk.Button(button_frame, text="拍攝照片")
btn_capture.pack(side="left", padx=10)

btn_flip = tk.Button(button_frame, text="翻轉照片")
btn_flip.pack(side="left", padx=10)

btn_contrast = tk.Button(button_frame, text="人臉對比度增強")
btn_contrast.pack(side="left", padx=10)

btn_blur = tk.Button(button_frame, text="人臉模糊化")
btn_blur.pack(side="left", padx=10)

btn_denoise = tk.Button(button_frame, text="影像雜訊去除")
btn_denoise.pack(side="left", padx=10)

btn_erode = tk.Button(button_frame, text="侵蝕")
btn_erode.pack(side="left", padx=10)

btn_dilate = tk.Button(button_frame, text="膨脹")
btn_dilate.pack(side="left", padx=10)

btn_track = tk.Button(button_frame, text="人臉追蹤")
btn_track.pack(side="left", padx=10)

btn_landmarks = tk.Button(button_frame, text="人臉特徵點檢測")
btn_landmarks.pack(side="left", padx=10)

btn_gray = tk.Button(button_frame, text="影像灰階")
btn_gray.pack(side="left", padx=10)

btn_edge = tk.Button(button_frame, text="邊緣檢測")
btn_edge.pack(side="left", padx=10)

btn_threshold = tk.Button(button_frame, text="二值化")
btn_threshold.pack(side="left", padx=10)

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
landmark_predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
flip_enabled = False
contrast_enabled = False
blur_enabled = False
denoise_enabled = False
erode_enabled = False
dilate_enabled = False
tracking_enabled = False
landmarks_enabled = False
gray_enabled = False
edge_enabled = False
threshold_enabled = False
tracker = dlib.correlation_tracker()
img = None

def face_detection():
    global img
    
    ret, frame = cap.read()
    
    if tracking_enabled:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        dlib_rect = dlib.rectangle(0, 0, frame.shape[1], frame.shape[0])
        tracker.start_track(frame_rgb, dlib_rect)
        tracker.update(frame_rgb)
        pos = tracker.get_position()
        x, y, w, h = int(pos.left()), int(pos.top()), int(pos.width()), int(pos.height())
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
    else:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
    
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            if contrast_enabled:
                roi = frame[y:y+h, x:x+w]
                lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)
                lab_planes = cv2.split(lab)
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                lab_planes_list = list(lab_planes)
                lab_planes_list[0] = clahe.apply(lab_planes_list[0])
                lab_planes = tuple(lab_planes_list)
                lab = cv2.merge(lab_planes)
                contrast_img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
                frame[y:y+h, x:x+w] = contrast_img
            
            if blur_enabled:
                roi = frame[y:y+h, x:x+w]
                blurred_img = cv2.GaussianBlur(roi, (99, 99), 0)
                frame[y:y+h, x:x+w] = blurred_img

            if denoise_enabled:
                roi = frame[y:y+h, x:x+w]
                denoised_img = cv2.fastNlMeansDenoisingColored(roi, None, 10, 10, 7, 21)
                frame[y:y+h, x:x+w] = denoised_img

            if erode_enabled:
                roi = frame[y:y+h, x:x+w]
                kernel = np.ones((5, 5), np.uint8)
                eroded_img = cv2.erode(roi, kernel, iterations=1)
                frame[y:y+h, x:x+w] = eroded_img

            if dilate_enabled:
                roi = frame[y:y+h, x:x+w]
                kernel = np.ones((5, 5), np.uint8)
                dilated_img = cv2.dilate(roi, kernel, iterations=1)
                frame[y:y+h, x:x+w] = dilated_img
            
            if landmarks_enabled:
                rect = dlib.rectangle(x, y, x+w, y+h)
                landmarks = landmark_predictor(gray, rect)
                for n in range(68):
                    x_landmark = landmarks.part(n).x
                    y_landmark = landmarks.part(n).y
                    cv2.circle(frame, (x_landmark, y_landmark), 1, (0, 0, 255), -1)

        if gray_enabled:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
        if edge_enabled:
            frame = cv2.Canny(frame, 100, 200)

        if threshold_enabled:
            _, frame = cv2.threshold(frame, 127, 255, cv2.THRESH_BINARY)
    
    if flip_enabled:
        frame = cv2.flip(frame, 1)
    
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)
    img = ImageTk.PhotoImage(image=img)
    
    label.imgtk = img
    label.configure(image=img)
    
    window.after(10, face_detection)

def capture_photo():
    ret, frame = cap.read()
    
    processed_frame = frame.copy()
    
    if contrast_enabled:
        lab = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2LAB)
        lab_planes = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        lab_planes_list = list(lab_planes)
        lab_planes_list[0] = clahe.apply(lab_planes_list[0])
        lab_planes = tuple(lab_planes_list)
        lab = cv2.merge(lab_planes)
        processed_frame = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    
    if blur_enabled:
        processed_frame = cv2.GaussianBlur(processed_frame, (99, 99), 0)
    
    if denoise_enabled:
        processed_frame = cv2.fastNlMeansDenoisingColored(processed_frame, None, 10, 10, 7, 21)
    
    if erode_enabled:
        kernel = np.ones((5, 5), np.uint8)
        processed_frame = cv2.erode(processed_frame, kernel, iterations=1)
    
    if dilate_enabled:
        kernel = np.ones((5, 5), np.uint8)
        processed_frame = cv2.dilate(processed_frame, kernel, iterations=1)
    
    if landmarks_enabled:
        gray = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
        
        for (x, y, w, h) in faces:
            rect = dlib.rectangle(x, y, x+w, y+h)
            landmarks = landmark_predictor(gray, rect)
            for n in range(68):
                x_landmark = landmarks.part(n).x
                y_landmark = landmarks.part(n).y
                cv2.circle(processed_frame, (x_landmark, y_landmark), 1, (0, 0, 255), -1)

    if gray_enabled:
        processed_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2GRAY)
    
    if edge_enabled:
        processed_frame = cv2.Canny(processed_frame, 100, 200)

    if threshold_enabled:
        _, processed_frame = cv2.threshold(processed_frame, 127, 255, cv2.THRESH_BINARY)
    
    cv2.imwrite("captured_photo.jpg", processed_frame)

def toggle_flip():
    global flip_enabled
    flip_enabled = not flip_enabled

def toggle_contrast():
    global contrast_enabled
    contrast_enabled = not contrast_enabled

def toggle_blur():
    global blur_enabled
    blur_enabled = not blur_enabled

def toggle_denoise():
    global denoise_enabled
    denoise_enabled = not denoise_enabled

def toggle_erode():
    global erode_enabled
    erode_enabled = not erode_enabled

def toggle_dilate():
    global dilate_enabled
    dilate_enabled = not dilate_enabled

def toggle_tracking():
    global tracking_enabled
    tracking_enabled = not tracking_enabled

def toggle_landmarks():
    global landmarks_enabled
    landmarks_enabled = not landmarks_enabled

def toggle_gray():
    global gray_enabled
    gray_enabled = not gray_enabled

def toggle_edge():
    global edge_enabled
    edge_enabled = not edge_enabled

def toggle_threshold():
    global threshold_enabled
    threshold_enabled = not threshold_enabled

btn_capture.config(command=capture_photo)
btn_flip.config(command=toggle_flip)
btn_contrast.config(command=toggle_contrast)
btn_blur.config(command=toggle_blur)
btn_denoise.config(command=toggle_denoise)
btn_erode.config(command=toggle_erode)
btn_dilate.config(command=toggle_dilate)
btn_track.config(command=toggle_tracking)
btn_landmarks.config(command=toggle_landmarks)
btn_gray.config(command=toggle_gray)
btn_edge.config(command=toggle_edge)
btn_threshold.config(command=toggle_threshold)

face_detection()

window.mainloop()
