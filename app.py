import base64
import io
import numpy as np
from PIL import Image

# Inisialisasi MediaPipe di luar route (biar cuma load 1x)
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# ... (Kode User, PostureLog, Login, Dashboard, dll tetap sama seperti sebelumnya) ...

# TAMBAHKAN ROUTE INI DI app.py LU:
@app.route('/api/analyze_posture', methods=['POST'])
@login_required
def api_analyze_posture():
    try:
        data = request.json
        image_data = data['image'] # Base64 image dari browser
        
        # Decode gambar
        header, encoded = image_data.split(",", 1)
        binary_data = base64.b64decode(encoded)
        image = Image.open(io.BytesIO(binary_data))
        image_np = np.array(image)
        
        # Proses dengan MediaPipe
        image_rgb = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR) # Sesuaikan format
        # MediaPipe butuh RGB
        image_rgb_mp = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb_mp)
        
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            x_bahu = (landmarks[11].x + landmarks[12].x) / 2
            y_bahu = (landmarks[11].y + landmarks[12].y) / 2
            x_pinggul = (landmarks[23].x + landmarks[24].x) / 2
            y_pinggul = (landmarks[23].y + landmarks[24].y) / 2

            dx = abs(x_bahu - x_pinggul)
            dy = y_pinggul - y_bahu 
            sudut = math.degrees(math.atan2(dx, dy))

            if sudut <= 5:
                skor = 1; kategori = "0 deg flexion"
            elif 5 < sudut <= 20:
                skor = 2; kategori = "0-20 deg flexion"
            elif 20 < sudut <= 60:
                skor = 3; kategori = "20-60 deg flexion"
            else:
                skor = 4; kategori = ">60 deg flexion"

            # Simpan ke database (opsional, bisa dikurangi frekuensinya)
            status_db = "Baik" if skor <= 2 else "Bungkuk"
            # log = PostureLog(user_id=current_user.id, status=status_db)
            # db.session.add(log)
            # db.session.commit()

            return jsonify({
                'success': True,
                'sudut': round(sudut, 1),
                'skor': skor,
                'kategori': kategori,
                'status': 'TEGAK' if skor <= 2 else 'BUNGKUK'
            })
        else:
            return jsonify({'success': False, 'message': 'Pose tidak terdeteksi'})
            
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})