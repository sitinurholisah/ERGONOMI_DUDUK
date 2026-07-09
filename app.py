import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import math
import time
import base64
import io

from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, send_file
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
import cv2
import numpy as np
from PIL import Image
import mediapipe as mp
from datetime import datetime, timedelta

# Inisialisasi Flask App
app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'rahasia-ergo-ai-2026')
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///ergonomi.db'
db = SQLAlchemy(app)

login_manager = LoginManager(app)
login_manager.login_view = 'login'

# Inisialisasi MediaPipe (HARUS setelah import)
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# ==================== VARIABEL GLOBAL ====================
system_status = "OFFLINE"
last_activity_time = None

# ==================== DATABASE MODELS ====================
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

# ==================== FUNGSI BANTUAN ====================
def hitung_skor_punggung(landmarks):
    x_bahu = (landmarks[11].x + landmarks[12].x) / 2
    y_bahu = (landmarks[11].y + landmarks[12].y) / 2
    x_pinggul = (landmarks[23].x + landmarks[24].x) / 2
    y_pinggul = (landmarks[23].y + landmarks[24].y) / 2

    dx = abs(x_bahu - x_pinggul)
    dy = y_pinggul - y_bahu 
    sudut = math.degrees(math.atan2(dx, dy))

    if sudut < 20:
        skor, kategori = 1, "0° – 20° flexion"
    elif sudut < 60:
        skor, kategori = 2, "20° - 60° flexion"
    else:
        skor, kategori = 3, "> 60° flexion"

    return sudut, skor, kategori

# ==================== ROUTES ====================
@app.route('/')
def splash():
    return redirect(url_for('login'))

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
    global system_status, last_activity_time
    
    if system_status == "ACTIVE" and last_activity_time:
        time_since_last = (datetime.now() - last_activity_time).total_seconds()
        if time_since_last > 30:
            system_status = "OFFLINE"
    
    if system_status == "ACTIVE":
        sys_status = "ACTIVE"
        sys_color = "#10b981"
        sys_icon = "fa-check-circle"
        sys_text = "Service Running"
    else:
        sys_status = "OFFLINE"
        sys_color = "#ef4444"
        sys_icon = "fa-times-circle"
        sys_text = "Service Interrupted"

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
                           sys_icon=sys_icon,
                           sys_text=sys_text)

@app.route('/api/analyze_posture', methods=['POST'])
@login_required
def api_analyze_posture():
    global system_status, last_activity_time
    
    try:
        system_status = "ACTIVE"
        last_activity_time = datetime.now()
        
        data = request.json
        image_data = data['image']
        
        header, encoded = image_data.split(",", 1)
        binary_data = base64.b64decode(encoded)
        image = Image.open(io.BytesIO(binary_data))
        image_np = np.array(image)
        
        image_rgb_mp = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb_mp)
        
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            sudut, skor, kategori = hitung_skor_punggung(landmarks)

            status_db = "Baik" if skor <= 1 else "Bungkuk"
            log = PostureLog(user_id=current_user.id, status=status_db)
            db.session.add(log)
            db.session.commit()

            return jsonify({
                'success': True,
                'sudut': round(sudut, 1),
                'skor': skor,
                'kategori': kategori,
                'status': 'TEGAK' if skor <= 1 else 'BUNGKUK'
            })
        else:
            return jsonify({'success': False, 'message': 'Pose tidak terdeteksi'})
            
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

@app.route('/api/log_posture', methods=['POST'])
@login_required
def api_log_posture():
    try:
        global system_status, last_activity_time
        system_status = "ACTIVE"
        last_activity_time = datetime.now()
        
        data = request.json
        skor = int(data['skor'])
        kategori = data['kategori']
        status_db = "Baik" if skor <= 1 else "Bungkuk"
        
        log = PostureLog(user_id=current_user.id, status=status_db)
        db.session.add(log)
        db.session.commit()
        
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

@app.route('/api_system_status')
@login_required
def api_system_status():
    global system_status
    return jsonify({
        'is_active': system_status == "ACTIVE",
        'status_text': system_status
    })

@app.route('/monitoring')
@login_required
def monitoring():
    return render_template('monitoring.html', nama=current_user.nama)

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
    pdf.cell(0, 8, f"Tanggal: {datetime.now().strftime('%d/%m/%Y')}", ln=True)
    pdf.ln(10)
    pdf.set_font("Arial", "B", 14)
    pdf.cell(0, 10, "STATISTIK 7 HARI", ln=True)
    today = datetime.now().date()
    total = 0
    for i in range(6, -1, -1):
        day = today - timedelta(days=i)
        count = PostureLog.query.filter(
            PostureLog.user_id == current_user.id,
            PostureLog.status == 'Bungkuk',
            db.func.date(PostureLog.waktu) == day
        ).count()
        total += count
        pdf.cell(0, 7, f"{day.strftime('%d/%m/%Y')}: {count}", ln=True)
    pdf.ln(10)
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 8, f"TOTAL: {total}", ln=True)
    filename = f"Laporan_{current_user.nama}.pdf"
    pdf.output(filename)
    return send_file(filename, as_attachment=True)

@app.route('/saran')
@login_required
def saran():
    return render_template('saran.html', nama=current_user.nama)

@app.route('/profil', methods=['GET', 'POST'])
@login_required
def profil():
    if request.method == 'POST':
        current_user.nama = request.form['nama']
        db.session.commit()
        flash('Profil diperbarui!', 'success')
        return redirect(url_for('profil'))
    return render_template('profil.html', nama=current_user.nama)

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))

# ==================== MAIN ====================
if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    
    port = int(os.environ.get('PORT', 5000))
    print(f"🚀 Starting ErgoVision on port {port}...")
    app.run(host='0.0.0.0', port=port, debug=True, threaded=True)