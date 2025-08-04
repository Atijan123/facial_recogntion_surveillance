from flask import Flask, render_template, request, Response, jsonify, redirect, url_for
from models import db, User, Log
from camera import Camera
from recognizer import FaceRecognizer
from werkzeug.utils import secure_filename
import os
from datetime import datetime
import cv2

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['UPLOAD_FOLDER'] = 'static/faces'
app.config['LOGS_FOLDER'] = 'static/logs'

# Ensure upload directories exist
for folder in [app.config['UPLOAD_FOLDER'], app.config['LOGS_FOLDER']]:
    if not os.path.exists(folder):
        os.makedirs(folder)

db.init_app(app)
with app.app_context():
    db.create_all()

camera = None
recognizer = FaceRecognizer(faces_dir=app.config['UPLOAD_FOLDER'])

def get_camera():
    global camera
    if camera is None:
        camera = Camera(faces_dir=app.config['UPLOAD_FOLDER'])
    return camera

def gen_frames():
    camera = get_camera()
    while True:
        frame, detected_faces = camera.get_frame()
        if frame is None:
            break

        for face in detected_faces:
            snapshot_path = camera.save_snapshot(face['frame'], logs_dir=app.config['LOGS_FOLDER'])
            log = Log(
                name=face['name'],
                snapshot_path=snapshot_path,
                confidence=face['confidence']
            )
            with app.app_context():
                db.session.add(log)
                db.session.commit()

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        if 'face' not in request.files:
            return 'No file uploaded', 400
            
        file = request.files['face']
        name = request.form.get('name')
        
        if file.filename == '':
            return 'No file selected', 400
            
        if name and file:
            filename = secure_filename(f"{name}.jpg")
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            if recognizer.register_face(filepath, name):
                user = User(name=name, image_path=filepath)
                db.session.add(user)
                db.session.commit()
                return redirect(url_for('register'))
            else:
                os.remove(filepath)
                return 'Face registration failed', 400
            
    return render_template('register.html')

@app.route('/scan', methods=['GET', 'POST'])
def scan():
    if request.method == 'POST':
        if 'image' not in request.files:
            return 'No file uploaded', 400
            
        file = request.files['image']
        if file.filename == '':
            return 'No file selected', 400

        filepath = recognizer.save_upload(file, app.config['LOGS_FOLDER'])
        name, confidence = recognizer.scan_image(filepath)

        log = Log(
            name=name if name else 'Unknown',
            snapshot_path=filepath,
            confidence=confidence
        )
        db.session.add(log)
        db.session.commit()

        return render_template('scan.html', 
                             result={'name': name if name else 'Unknown',
                                    'confidence': confidence,
                                    'image_path': filepath})

    return render_template('scan.html')

@app.route('/logs')
def logs():
    search = request.args.get('search', '')
    query = Log.query
    
    if search:
        query = query.filter(Log.name.ilike(f'%{search}%'))
        
    logs = query.order_by(Log.timestamp.desc()).all()
    return render_template('logs.html', logs=logs, search=search)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port = 5000)