import cv2
import mediapipe as mp
import numpy as np
import os
import time

print("=== PENGUMPUL DATASET POSTUR ===")
print("Tekan:")
print("  't' = Postur TEGAK")
print("  'b' = Postur BUNGKUK")
print("  'm' = Postur MIRING")
print("  'q' = Keluar")

# Setup MediaPipe
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Buka Webcam
cap = cv2.VideoCapture(0)

# Folder dataset
dataset_dir = 'dataset'
os.makedirs(dataset_dir, exist_ok=True)

# Counter untuk setiap kelas
counters = {'tegak': 0, 'bungkuk': 0, 'miring': 0}

# Load counter yang udah ada
for label in ['tegak', 'bungkuk', 'miring']:
    folder_path = os.path.join(dataset_dir, label)
    if os.path.exists(folder_path):
        counters[label] = len(os.listdir(folder_path))

current_label = None
recording = False
start_time = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Flip frame biar kayak ngaca
    frame = cv2.flip(frame, 1)
    
    # Proses MediaPipe
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = pose.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    # Gambar landmark
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        
        # Kalau lagi merekam, simpan data
        if recording and current_label:
            landmarks = results.pose_landmarks.landmark
            # Ambil koordinat x, y dari semua landmark (33 titik)
            coords = []
            for lm in landmarks:
                coords.extend([lm.x, lm.y, lm.z])
            
            # Simpan ke file numpy
            folder_path = os.path.join(dataset_dir, current_label)
            os.makedirs(folder_path, exist_ok=True)
            
            filename = os.path.join(folder_path, f"{counters[current_label]}.npy")
            np.save(filename, coords)
            counters[current_label] += 1
            
            # Stop setelah 3 detik (sekitar 90 frame)
            if time.time() - start_time > 3:
                recording = False
                current_label = None
                print(f"✓ Data {current_label} tersimpan!")
    
    # Tampilkan info di layar
    cv2.putText(image, f'TEGAK: {counters["tegak"]}', (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(image, f'BUNGKUK: {counters["bungkuk"]}', (10, 60), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.putText(image, f'MIRING: {counters["miring"]}', (10, 90), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    
    if recording:
        cv2.putText(image, f'RECORDING: {current_label.upper()}', (10, 120), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 3)
    
    cv2.imshow('Collect Data - Tekan t/b/m untuk merekam', image)
    
    key = cv2.waitKey(10) & 0xFF
    
    if key == ord('q'):
        break
    elif key == ord('t') and not recording:
        current_label = 'tegak'
        recording = True
        start_time = time.time()
        print(f"▶ Mulai merekam TEGAK...")
    elif key == ord('b') and not recording:
        current_label = 'bungkuk'
        recording = True
        start_time = time.time()
        print(f"▶ Mulai merekam BUNGKUK...")
    elif key == ord('m') and not recording:
        current_label = 'miring'
        recording = True
        start_time = time.time()
        print(f"▶ Mulai merekam MIRING...")

cap.release()
cv2.destroyAllWindows()

print("\n=== DATASET LENGKAP ===")
for label, count in counters.items():
    print(f"{label.upper()}: {count} data")
print(f"Total: {sum(counters.values())} data")