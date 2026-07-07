import cv2
import mediapipe as mp
import numpy as np
import csv
import os
import io
from datetime import datetime
from flask import Flask, render_template, Response, request, jsonify, send_file
from fpdf import FPDF

# Inisialisasi aplikasi Flask
app = Flask(__name__)

# Buat folder 'hasil' kalau belum ada untuk menyimpan data lokal sementara
os.makedirs('hasil', exist_ok=True)

# Inisialisasi MediaPipe Pose
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Variabel global untuk menyimpan data log deteksi di memori server
log_data = []

def calculate_angle(a, b, c):
    """Fungsi untuk menghitung sudut antara tiga titik koordinat tubuh."""
    a = np.array([a.x, a.y]) # Titik pertama (misal: Telinga)
    b = np.array([b.x, b.y]) # Titik tengah/sudut (misal: Bahu)
    c = np.array([c.x, c.y]) # Titik ketiga (misal: Pinggang)
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    
    if angle > 180.0:
        angle = 360.0 - angle
    return angle

@app.route('/')
def index():
    """Halaman utama website aplikasi ergonomi duduk."""
    return """
    <!DOCTYPE html>
    <html lang="id">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Sistem Deteksi Ergonomi Duduk</title>
        <style>
            body { font-family: Arial, sans-serif; text-align: center; background-color: #f4f4f9; padding: 20px; }
            h1 { color: #333; }
            .container { display: flex; flex-direction: column; align-items: center; justify-content: center; margin-top: 20px; }
            video { border: 4px solid #333; border-radius: 8px; width: 100%; max-width: 640px; transform: scaleX(-1); }
            canvas { display: none; }
            .status-box { margin-top: 20px; padding: 15px; font-size: 24px; font-weight: bold; border-radius: 8px; width: 300px; color: white; }
            .status-baik { background-color: #28a745; }
            .status-warning { background-color: #ffc107; color: #333; }
            .status-bad { background-color: #dc3545; }
            .btn { margin-top: 20px; padding: 10px 20px; font-size: 16px; background-color: #007bff; color: white; border: none; border-radius: 5px; cursor: pointer; text-decoration: none; display: inline-block; }
            .btn:hover { background-color: #0056b3; }
        </style>
    </head>
    <body>
        <h1>Sistem Monitoring Ergonomi Positural Duduk</h1>
        <div class="container">
            <video id="webcam" autoplay playsinline></video>
            <canvas id="canvasFrame" width="640" height="480"></canvas>
            <div id="statusDisplay" class="status-box status-baik">Menginisialisasi...</div>
            <a href="/download-pdf" class="btn">Unduh Laporan PDF Hasil Deteksi</a>
        </div>

        <script>
            const video = document.getElementById('webcam');
            const canvas = document.getElementById('canvasFrame');
            const context = canvas.getContext('2d');
            const statusDisplay = document.getElementById('statusDisplay');

            // Akses kamera pengguna dari sisi browser client
            navigator.mediaDevices.getUserMedia({ video: true })
                .then(stream => { video.srcObject = stream; })
                .catch(err => { console.error("Gagal mengakses kamera: ", err); alert("Harap izinkan akses kamera!"); });

            // Kirim frame gambar secara berkala ke backend server Railway tiap 200ms
            setInterval(() => {
                if (video.readyState === video.HAVE_ENOUGH_DATA) {
                    context.drawImage(video, 0, 0, 640, 480);
                    const dataUrl = canvas.toDataURL('image/jpeg', 0.7);
                    
                    fetch('/process-frame', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ image: dataUrl })
                    })
                    .then(response => response.json())
                    .then(data => {
                        statusDisplay.innerText = `Sudut: ${data.angle}° | ${data.status}`;
                        statusDisplay.className = "status-box";
                        
                        if (data.status === "AWAS BUNGKUK!") {
                            statusDisplay.classList.add("status-bad");
                            // Browser client membunyikan alarm audio (sebagai ganti winsound di server)
                            const audioCtx = new (window.AudioContext || window.webkitAudioContext)();
                            const oscillator = audioCtx.createOscillator();
                            oscillator.type = 'sine';
                            oscillator.frequency.setValueAtTime(1000, audioCtx.currentTime);
                            oscillator.connect(audioCtx.destination);
                            oscillator.start();
                            oscillator.stop(audioCtx.currentTime + 0.3);
                        } else if (data.status === "Hampir Tegak") {
                            statusDisplay.classList.add("status-warning");
                        } else {
                            statusDisplay.classList.add("status-baik");
                        }
                    })
                    .catch(err => console.error("Eror memproses frame:", err));
                }
            }, 200);
        </script>
    </body>
    </html>
    """

@app.route('/process-frame',延期=['POST'])
@app.route('/process-frame', methods=['POST'])
def process_frame():
    """Endpoint untuk menerima dan menganalisis gambar dari browser."""
    global log_data
    data = request.get_json()
    if not data or 'image' not in data:
        return jsonify({'angle': 0, 'status': 'Tidak ada frame'})

    # Ekstrak data base64 gambar
    image_data = data['image'].split(',')[1]
    import base64
    image_bytes = base64.b64decode(image_data)
    
    # Konversi bytes ke bentuk matriks citra OpenCV
    nparr = np.frombuffer(image_bytes, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if frame is None:
        return jsonify({'angle': 0, 'status': 'Gagal membaca gambar'})

    # Pemrosesan MediaPipe
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image_rgb.flags.writeable = False
    results = pose.process(image_rgb)
    
    posture_status = "Postur Baik"
    avg_angle = 0
    
    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark
        
        left_ear = landmarks[mp_pose.PoseLandmark.LEFT_EAR.value]
        left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
        left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
        
        right_ear = landmarks[mp_pose.PoseLandmark.RIGHT_EAR.value]
        right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
        right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]
        
        left_angle = calculate_angle(left_ear, left_shoulder, left_hip)
        right_angle = calculate_angle(right_ear, right_shoulder, right_hip)
        avg_angle = int((left_angle + right_angle) / 2)
        
        if avg_angle < 160:  
            posture_status = "AWAS BUNGKUK!"
        elif avg_angle < 170:
            posture_status = "Hampir Tegak"
        else:
            posture_status = "Postur Baik"
            
        # Simpan rekam medis ergonomi ke memori data
        waktu_sekarang = datetime.now().strftime("%H:%M:%S")
        log_data.append([waktu_sekarang, str(avg_angle), posture_status])

    return jsonify({'angle': avg_angle, 'status': posture_status})

@app.route('/download-pdf')
def download_pdf():
    """Endpoint untuk membuat dan mengunduh laporan PDF dari library fpdf2."""
    global log_data
    
    pdf = FPDF(orientation='P', unit='mm', format='A4')
    pdf.add_page()
    pdf.set_font("Arial", style='B', size=16)
    pdf.cell(190, 10, txt="LAPORAN ANALISIS ERGONOMI POSTUR DUDUK", ln=1, align="C")
    pdf.set_font("Arial", size=10)
    pdf.cell(190, 10, txt=f"Tanggal Cetak: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=1, align="C")
    pdf.ln(10)
    
    # Header Tabel
    pdf.set_font("Arial", style='B', size=11)
    pdf.cell(45, 10, "Waktu", border=1, align="C")
    pdf.cell(45, 10, "Sudut Tulang Belakang", border=1, align="C")
    pdf.cell(100, 10, "Kategori Evaluasi", border=1, align="C")
    pdf.ln()
    
    # Isi Tabel dari rekaman data terbaru (maksimal 25 baris terakhir biar rapi)
    pdf.set_font("Arial", size=10)
    for row in log_data[-25:]:
        pdf.cell(45, 8, row[0], border=1, align="C")
        pdf.cell(45, 8, f"{row[1]} derajat", border=1, align="C")
        pdf.cell(100, 8, row[2], border=1, align="C")
        pdf.ln()
        
    # Menyimpan output pdf langsung ke memori untuk diunduh client browser
    pdf_output = pdf.output(dest='S').encode('latin-1')
    buffer = io.BytesIO(pdf_output)
    buffer.seek(0)
    
    return send_file(
        buffer,
        mimetype='application/pdf',
        as_attachment=True,
        download_name=f"Laporan_Ergonomi_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
    )

if __name__ == '__main__':
    # Railway mengharuskan aplikasi mendengar ke Port lingkungan dinamis yang mereka sediakan
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)