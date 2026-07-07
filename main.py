import cv2
import mediapipe as mp
import numpy as np
import winsound
import csv
import os
from datetime import datetime

print("Membuka webcam...")

# Buat folder 'hasil' kalau belum ada
os.makedirs('hasil', exist_ok=True)

# Inisialisasi MediaPipe Pose
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Buka Webcam
cap = cv2.VideoCapture(0)

# Variabel untuk nyimpan data
log_data = []
beep_cooldown = 0  # Biar beep nggak bunyi terus-terusan tiap frame

def calculate_angle(a, b, c):
    a = np.array([a.x, a.y])
    b = np.array([b.x, b.y])
    c = np.array([c.x, c.y])
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0: angle = 360 - angle
    return angle

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break

    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = pose.process(image)

    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    posture_status = "Postur Baik"
    posture_color = (0, 255, 0)  # Hijau
    avg_angle = 0
    
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        landmarks = results.pose_landmarks.landmark
        
        left_ear = landmarks[mp_pose.PoseLandmark.LEFT_EAR.value]
        left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
        left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
        
        right_ear = landmarks[mp_pose.PoseLandmark.RIGHT_EAR.value]
        right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
        right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]
        
        left_angle = calculate_angle(left_ear, left_shoulder, left_hip)
        right_angle = calculate_angle(right_ear, right_shoulder, right_hip)
        avg_angle = (left_angle + right_angle) / 2
        
        # Deteksi postur
        if avg_angle < 160:  
            posture_status = "AWAS BUNGKUK!"
            posture_color = (0, 0, 255)  # Merah
            
            # Bunyikan beep (cooldown 3 detik / 90 frame)
            if beep_cooldown <= 0:
                winsound.Beep(1000, 500) # Frekuensi 1000Hz, durasi 500ms
                beep_cooldown = 90 
        elif avg_angle < 170:
            posture_status = "Hampir Tegak"
            posture_color = (0, 255, 255)  # Kuning
        else:
            posture_status = "Postur Baik"
            posture_color = (0, 255, 0)  # Hijau
            
        # Kurangi cooldown beep
        if beep_cooldown > 0:
            beep_cooldown -= 1

        # Tampilkan teks
        cv2.rectangle(image, (0, 0), (450, 100), (0, 0, 0), -1)
        cv2.putText(image, f'Sudut: {int(avg_angle)}° | {posture_status}', 
                   (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, posture_color, 2)
        
        # Simpan data ke memory (tiap 30 frame / 1 detik)
        if len(log_data) == 0 or (len(log_data) > 0 and (len(log_data) % 30 == 0)):
            log_data.append([datetime.now().strftime("%H:%M:%S"), f"{int(avg_angle)}", posture_status])

    cv2.imshow('Ergonomi Duduk - Deteksi Postur', image)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

# Matikan webcam
cap.release()
cv2.destroyAllWindows()

# Simpan data ke CSV
if log_data:
    filename = f"hasil/laporan_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Waktu", "Sudut_Badan", "Status_Postur"])
        writer.writerows(log_data)
    print(f"Laporan berhasil disimpan di: {filename}")
else:
    print("Tidak ada data yang direkam.")

print("Program selesai.")