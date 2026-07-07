import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import math
import time

from flask import Flask, render_template, request, redirect, url_for, flash, Response, jsonify, send_file
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
import cv2
import mediapipe as mp
from datetime import datetime, timedelta

app = Flask(__name__)
app.config['SECRET_KEY'] = 'rahasia-ergo-ai-2026'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///ergonomi.db'
db = SQLAlchemy(app)

login_manager = LoginManager(app)
login_manager.login_view = 'login'

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    nama = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(100), unique=True, nullable=False)
    password = db.Column(db.String(250), nullable=False)

class PostureLog(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    status = db.Column(db.String(50), nullable=False)
    waktu = db.Column(db.DateTime, default=datetime.utcnow)

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# --- VARIABEL GLOBAL UNTUK KONTROL KAMERA ---
camera = None
is_streaming = False  # Ini saklarnya. True = Nyala, False = Mati
system_status = "OFFLINE" # Status real-time
current_ai_status = {"status": "Mencari...", "is_bad": False}
frame_count = 0
current_monitoring_user_id = None

def initialize_camera():
    global camera
    if camera is not None:
        camera.release()
        time.sleep(0.5)
    
    print("🔄 Initializing camera...")
    camera = cv2.VideoCapture(0)
    time.sleep(1)
    
    if camera.isOpened():
        success, test_frame = camera.read()
        if success and test_frame is not None:
            print("✅ Camera initialized successfully!")
            return True
        else:
            print("❌ Camera opened but cannot read frame")
            camera.release()
            camera = None
            return False
    else:
        print("❌ Camera failed to open")
        camera = None
        return False

def hitung_skor_punggung_tabel5(landmarks):
    x_bahu = (landmarks[11].x + landmarks[12].x) / 2
    y_bahu = (landmarks[11].y + landmarks[12].y) / 2
    x_pinggul = (landmarks[23].x + landmarks[24].x) / 2
    y_pinggul = (landmarks[23].y + landmarks[24].y) / 2

    dx = abs(x_bahu - x_pinggul)
    dy = y_pinggul - y_bahu 
    sudut = math.degrees(math.atan2(dx, dy))

    if sudut <= 5:
        skor = 1
        kategori = "0 deg flexion"
    elif 5 < sudut <= 20:
        skor = 2
        kategori = "0-20 deg flexion"
    elif 20 < sudut <= 60:
        skor = 3
        kategori = "20-60 deg flexion"
    else:
        skor = 4
        kategori = ">60 deg flexion"

    return sudut, skor, kategori

def generate_frames():
    global frame_count, current_monitoring_user_id, camera, is_streaming, system_status
    
    is_streaming = True # Nyalakan saklar
    
    if camera is None or not camera.isOpened():
        if not initialize_camera():
            system_status = "OFFLINE"
            return
    
    system_status = "ACTIVE"

    while is_streaming: # Loop selama saklar nyala
        success, frame = camera.read()
        if not success:
            system_status = "OFFLINE"
            break
        
        frame = cv2.flip(frame, 1)
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = pose.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        current_status_db = "Baik"

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            
            x_bahu = (landmarks[11].x + landmarks[12].x) / 2
            y_bahu = (landmarks[11].y + landmarks[12].y) / 2
            x_pinggul = (landmarks[23].x + landmarks[24].x) / 2
            y_pinggul = (landmarks[23].y + landmarks[24].y) / 2
            
            img_height, img_width = image.shape[:2]
            pt_bahu = (int(x_bahu * img_width), int(y_bahu * img_height))
            pt_pinggul = (int(x_pinggul * img_width), int(y_pinggul * img_height))
            
            cv2.line(image, pt_bahu, pt_pinggul, (0, 255, 255), 4)
            cv2.circle(image, pt_bahu, 10, (0, 255, 0), -1)
            cv2.circle(image, pt_pinggul, 10, (0, 0, 255), -1)
            
            sudut, skor, kategori = hitung_skor_punggung_tabel5(landmarks)
            
            cv2.rectangle(image, (5, 5), (350, 110), (0, 0, 0), -1)
            cv2.putText(image, f"Sudut Punggung: {sudut:.1f} deg", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.putText(image, f"Skor Tabel 5: {skor}", (10, 60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(image, kategori, (10, 90), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            if skor <= 2:
                current_status_db = "Baik"
                current_ai_status["status"] = "TEGAK"
                current_ai_status["is_bad"] = False
            else:
                current_status_db = "Bungkuk"
                current_ai_status["status"] = "BUNGKUK"
                current_ai_status["is_bad"] = True
            
            frame_count += 1
            if frame_count % 90 == 0 and current_monitoring_user_id is not None:
                try:
                    with app.app_context():
                        log = PostureLog(user_id=current_monitoring_user_id, status=current_status_db)
                        db.session.add(log)
                        db.session.commit()
                except Exception as e:
                    print(f"DB Error: {e}")

        ret, buffer = cv2.imencode('.jpg', image)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    # Saat loop berhenti (karena is_streaming = False)
    is_streaming = False
    system_status = "OFFLINE"
    print(" Streaming loop ended.")

@app.route('/')
def splash():
    return render_template('splash.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        user = User.query.filter_by(email=email).first()
        if user and check_password_hash(user.password, password):
            login_user(user)
            return redirect(url_for('dashboard'))
        flash('Email atau password salah!', 'error')
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        nama = request.form['nama']
        email = request.form['email']
        password = request.form['password']
        hashed_pw = generate_password_hash(password)
        new_user = User(nama=nama, email=email, password=hashed_pw)
        try:
            db.session.add(new_user)
            db.session.commit()
            flash('Registrasi berhasil! Silakan login.', 'success')
            return redirect(url_for('login'))
        except:
            flash('Email sudah terdaftar!', 'error')
    return render_template('register.html')

@app.route('/dashboard')
@login_required
def dashboard():
    # Kita pakai variabel global system_status, gak perlu akses kamera langsung di sini
    # Ini mencegah error konflik akses kamera
    global system_status
    
    if system_status == "ACTIVE":
        sys_status = "ACTIVE"
        sys_color = "green"
        sys_icon = "fa-check-circle"
    else:
        sys_status = "OFFLINE / ERROR"
        sys_color = "red"
        sys_icon = "fa-times-circle"

    today = datetime.now().date()
    tanggal_hari_ini = datetime.now().strftime('%d %B %Y')

    warning_count = PostureLog.query.filter(
        PostureLog.user_id == current_user.id,
        PostureLog.status == 'Bungkuk',
        db.func.date(PostureLog.waktu) == today
    ).count()
    
    total_logs = PostureLog.query.filter(
        PostureLog.user_id == current_user.id,
        db.func.date(PostureLog.waktu) == today
    ).count()
    duration_seconds = total_logs * 3
    
    hours = duration_seconds // 3600
    minutes = (duration_seconds % 3600) // 60
    seconds = duration_seconds % 60
    duration_str = f"{hours:02d}:{minutes:02d}:{seconds:02d}"

    return render_template('dashboard.html', 
                           nama=current_user.nama, 
                           warning_count=warning_count, 
                           duration=duration_str,
                           tanggal_hari_ini=tanggal_hari_ini,
                           sys_status=sys_status,
                           sys_color=sys_color,
                           sys_icon=sys_icon)

@app.route('/api_system_status')
@login_required
def api_system_status():
    global system_status, camera
    return jsonify({
        'is_active': system_status == "ACTIVE",
        'status_text': system_status
    })

@app.route('/monitoring')
@login_required
def monitoring():
    global current_monitoring_user_id
    current_monitoring_user_id = current_user.id
    print("🎬 Monitoring route accessed...")
    return render_template('monitoring.html', nama=current_user.nama)

@app.route('/video_feed')
@login_required
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/stop_camera')
@login_required
def stop_camera():
    global is_streaming, camera, system_status, current_monitoring_user_id, frame_count
    
    print(" Stopping camera process...")
    is_streaming = False # Matikan saklar, loop di generate_frames bakal berhenti
    
    time.sleep(1) # Kasih waktu 1 detik biar proses video sempet berhenti total
    
    if camera is not None:
        camera.release()
        camera = None
    
    system_status = "OFFLINE"
    current_monitoring_user_id = None
    frame_count = 0
    print("✅ Camera fully released.")
    
    flash('Camera telah dimatikan.', 'info')
    return redirect(url_for('dashboard'))

@app.route('/statistik')
@login_required
def statistik():
    today = datetime.now().date()
    labels = []
    data = []
    for i in range(6, -1, -1):
        day = today - timedelta(days=i)
        labels.append(day.strftime("%d/%m"))
        count = PostureLog.query.filter(
            PostureLog.user_id == current_user.id,
            PostureLog.status == 'Bungkuk',
            db.func.date(PostureLog.waktu) == day
        ).count()
        data.append(count)
    return render_template('statistik.html', nama=current_user.nama, labels=labels, data=data)

@app.route('/riwayat')
@login_required
def riwayat():
    logs = PostureLog.query.filter_by(user_id=current_user.id).order_by(PostureLog.waktu.desc()).limit(50).all()
    return render_template('riwayat.html', nama=current_user.nama, logs=logs)

@app.route('/laporan')
@login_required
def laporan():
    return render_template('laporan.html', nama=current_user.nama)

@app.route('/download_laporan')
@login_required
def download_laporan():
    from fpdf import FPDF
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", "B", 20)
    pdf.cell(0, 15, "LAPORAN ERGOVISION", ln=True, align="C")
    pdf.ln(5)
    pdf.set_font("Arial", "", 12)
    pdf.cell(0, 8, f"Nama: {current_user.nama}", ln=True)
    pdf.cell(0, 8, f"Email: {current_user.email}", ln=True)
    pdf.cell(0, 8, f"Tanggal Cetak: {datetime.now().strftime('%d/%m/%Y %H:%M')}", ln=True)
    pdf.ln(10)
    pdf.set_font("Arial", "B", 14)
    pdf.cell(0, 10, "STATISTIK 7 HARI TERAKHIR", ln=True)
    pdf.ln(5)
    pdf.set_font("Arial", "", 11)
    today = datetime.now().date()
    total_bungkuk = 0
    for i in range(6, -1, -1):
        day = today - timedelta(days=i)
        count = PostureLog.query.filter(
            PostureLog.user_id == current_user.id,
            PostureLog.status == 'Bungkuk',
            db.func.date(PostureLog.waktu) == day
        ).count()
        total_bungkuk += count
        pdf.cell(0, 7, f"{day.strftime('%d/%m/%Y')}: {count} kali bungkuk", ln=True)
    pdf.ln(10)
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 8, f"TOTAL PERINGATAN: {total_bungkuk} kali", ln=True)
    pdf.ln(10)
    pdf.set_font("Arial", "B", 14)
    pdf.cell(0, 10, "SARAN ERGONOMI", ln=True)
    pdf.ln(5)
    pdf.set_font("Arial", "", 11)
    saran_list = [
        "1. Duduklah dengan punggung tegak dan bersandar pada sandaran kursi.",
        "2. Letakkan kaki rata di lantai, jangan menyilangkan kaki.",
        "3. Posisi layar monitor sejajar dengan mata (eye level).",
        "4. Istirahat setiap 30 menit untuk meregangkan otot.",
        "5. Lakukan peregangan leher dan bahu secara berkala."
    ]
    for saran in saran_list:
        pdf.cell(0, 7, saran, ln=True)
    pdf.ln(15)
    pdf.set_font("Arial", "I", 10)
    pdf.cell(0, 8, "Dokumen ini dibuat otomatis oleh sistem ERGOVISION.", ln=True, align="C")
    filename = f"Laporan_ErgoVision_{current_user.nama}_{today}.pdf"
    pdf.output(filename)
    return send_file(filename, as_attachment=True)

@app.route('/profil', methods=['GET', 'POST'])
@login_required
def profil():
    if request.method == 'POST':
        nama_baru = request.form['nama']
        current_user.nama = nama_baru
        db.session.commit()
        flash('Profil berhasil diperbarui!', 'success')
        return redirect(url_for('profil'))
    return render_template('profil.html', nama=current_user.nama)

@app.route('/saran')
@login_required
def saran():
    return render_template('saran.html', nama=current_user.nama)

@app.route('/pengaturan')
@login_required
def pengaturan():
    if request.method == 'POST':
        flash('Pengaturan berhasil disimpan!', 'success')
        return redirect(url_for('pengaturan'))
    return render_template('pengaturan.html', nama=current_user.nama)

@app.route('/api_status')
@login_required
def api_status():
    return jsonify(current_ai_status)

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True, port=5000, threaded=True)