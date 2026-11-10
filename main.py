import cv2
import mediapipe as mp
import numpy as np
import math
import csv
import time
from datetime import datetime
from pathlib import Path

# =====================================
# 1. ТОХИРГОО БА САНГУУД
# =====================================
MODEL_PATH = "pose_landmarker_lite.task"
TARGET_ANGLE = 95
STANDING_ANGLE = 160
VISIBILITY_LIMIT = 0.60

def calculate_angle(hip, knee, ankle):
    a, b, c = np.array(hip), np.array(knee), np.array(ankle)
    ba, bc = a - b, c - b
    denominator = np.linalg.norm(ba) * np.linalg.norm(bc)
    if denominator == 0: return None
    cosine = np.clip(np.dot(ba, bc) / denominator, -1.0, 1.0)
    return math.degrees(math.acos(cosine))

def get_knee_angle(landmarks, width, height):
    candidates = []
    for hip_id, knee_id, ankle_id in [(23, 25, 27), (24, 26, 28)]:
        hip, knee, ankle = landmarks[hip_id], landmarks[knee_id], landmarks[ankle_id]
        visibility = min(hip.visibility, knee.visibility, ankle.visibility)
        if visibility < VISIBILITY_LIMIT: continue
        hip_xy = (hip.x * width, hip.y * height)
        knee_xy = (knee.x * width, knee.y * height)
        ankle_xy = (ankle.x * width, ankle.y * height)
        angle = calculate_angle(hip_xy, knee_xy, ankle_xy)
        if angle is not None:
            candidates.append((visibility, angle, knee_xy))
    if not candidates: return None, None
    visibility, angle, knee_xy = max(candidates, key=lambda item: item[0])
    return angle, knee_xy

if not Path(MODEL_PATH).exists():
    raise FileNotFoundError("Model file not found: " + MODEL_PATH)

BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode

options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=RunningMode.VIDEO, num_poses=1,
    min_pose_detection_confidence=0.5, min_pose_presence_confidence=0.5,
    min_tracking_confidence=0.5
)

connections = [(11, 13), (13, 15), (12, 14), (14, 16), (11, 12), (11, 23), (12, 24), (23, 24), (23, 25), (25, 27), (24, 26), (26, 28)]
cap = cv2.VideoCapture(0)

reps = 0
score = 0  # ШИНЭ: Тоглоомын оноо
target_reached = False
start_time = time.monotonic()
previous_timestamp_ms = -1

# =====================================
# 2. ҮНДСЭН ДАВТАЛТ
# =====================================
try:
    with PoseLandmarker.create_from_options(options) as landmarker:
        while True:
            ret, frame = cap.read()
            if not ret: break
            frame = cv2.flip(frame, 1)
            height, width = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            
            elapsed_ms = int((time.monotonic() - start_time) * 1000)
            timestamp_ms = max(elapsed_ms, previous_timestamp_ms + 1)
            previous_timestamp_ms = timestamp_ms
            
            result = landmarker.detect_for_video(mp_image, timestamp_ms)
            angle, knee_xy, status = None, None, "Ready"
            progress = 0 # ШИНЭ: Progress bar-ийн хувь

            if result.pose_landmarks:
                landmarks = result.pose_landmarks[0]
                for start_id, end_id in connections:
                    p1, p2 = landmarks[start_id], landmarks[end_id]
                    if min(p1.visibility, p2.visibility) >= 0.5:
                        cv2.line(frame, (int(p1.x * width), int(p1.y * height)), (int(p2.x * width), int(p2.y * height)), (255, 255, 255), 2)
                angle, knee_xy = get_knee_angle(landmarks, width, height)
                
                if knee_xy is not None:
                    cv2.circle(frame, (int(knee_xy[0]), int(knee_xy[1])), 8, (0, 255, 255), -1)

            if angle is not None:
                # ШИНЭ: Өнцгийг 0-100% руу хөрвүүлэх (160 градус = 0%, 95 градус = 100%)
                progress = np.interp(angle, [TARGET_ANGLE, STANDING_ANGLE], [100, 0])
                
                if angle <= TARGET_ANGLE:
                    if not target_reached:
                        target_reached = True
                        score += 10 # Гүйцэд суувал 10 оноо
                    status = "PERFECT!"
                elif angle >= STANDING_ANGLE:
                    if target_reached:
                        reps += 1
                        target_reached = False
                        status = "+1 COMBO!"
                    else:
                        status = "Stand up straight"
                elif target_reached:
                    status = "Keep going up!"
                else:
                    status = "Squat down..."

            # =====================================
            # ШИНЭ: ТОГЛООМЫН ИНТЕРФЭЙС (HUD) ЗУРАХ
            # =====================================
            # 1. Бараан дэвсгэр
            cv2.rectangle(frame, (0, 0), (width, 80), (20, 20, 20), -1)
            
            # 2. Оноо болон Давталт харуулах
            cv2.putText(frame, f"REPS: {reps}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
            cv2.putText(frame, f"SCORE: {score}", (200, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 215, 255), 3)
            
            # 3. Төлөв байдал (PERFECT, COMBO гэх мэт)
            status_color = (0, 255, 0) if "PERFECT" in status or "COMBO" in status else (255, 255, 255)
            cv2.putText(frame, status, (width // 2 - 50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, status_color, 3)

            # 4. ДҮҮРДЭГ ХЭМЖҮҮР (PROGRESS BAR) ЗУРАХ
            bar_x1, bar_y1 = width - 80, 150
            bar_x2, bar_y2 = width - 40, 400
            
            # Гаднах хүрээ
            cv2.rectangle(frame, (bar_x1, bar_y1), (bar_x2, bar_y2), (100, 100, 100), 3)
            
            if angle is not None:
                # Өнгө тооцоолох (Улаанаас Ногоон руу)
                r = int(255 * (1 - progress / 100))
                g = int(255 * (progress / 100))
                color = (0, g, r) # OpenCV нь BGR ашигладаг (Цэнхэр, Ногоон, Улаан)
                
                # Доторх дүүрэх хэсэг
                fill_y = int(np.interp(progress, [0, 100], [bar_y2, bar_y1]))
                cv2.rectangle(frame, (bar_x1, fill_y), (bar_x2, bar_y2), color, cv2.FILLED)
                
                # Хувь харуулах
                cv2.putText(frame, f"{int(progress)}%", (bar_x1 - 10, bar_y2 + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            cv2.imshow("FitCam - Gamified Squat Tracker", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"): break

finally:
    cap.release()
    cv2.destroyAllWindows()
    print(f"Game Over! Total Reps: {reps} | Total Score: {score}")