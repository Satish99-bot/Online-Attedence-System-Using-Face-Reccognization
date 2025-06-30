from flask import Flask, render_template, Response, request, send_file, jsonify
import cv2
import numpy as np
import tensorflow as tf
from mtcnn import MTCNN
import os
import io
import time
from datetime import datetime
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from sklearn.preprocessing import LabelEncoder
from imutils.video import VideoStream
from keras_facenet import FaceNet
import pandas as pd
from collections import deque
from werkzeug.utils import secure_filename

# Initialize FaceNet and video stream
embedder = FaceNet()
facenet_model = embedder.model
vs = None



# Constants (maintained exactly from original)
TRAIN_FOLDER = "train_set"
MODEL_PATH = "resnet_model.keras"
LABEL_CLASSES_PATH = "label_encoder_classes.npy"
ATTENDANCE_FILE_PATH = 'attendance.xlsx'

# Detection parameters (unchanged from original)
MIN_FACE_SIZE = 64
BLUR_THRESHOLD = 50
CONFIDENCE_THRESHOLD = 0.80
ATTENDANCE_COOLDOWN = 5
FRAME_SKIP = 1

app = Flask(__name__)
app.secret_key = 'your_secret_key_here' 

# Load models (original initialization logic)
trained_model = load_model(MODEL_PATH)
detector = MTCNN()
label_encoder = LabelEncoder()
label_encoder.classes_ = np.load(LABEL_CLASSES_PATH)
session = {}
face_history = {}

# Original quality check function
def is_face_quality_good(face_image):
    try:
        if face_image.size == 0:
            return False
        face_image = cv2.resize(face_image, (160, 160))
        gray = cv2.cvtColor(face_image, cv2.COLOR_RGB2GRAY)
        fm = cv2.Laplacian(gray, cv2.CV_64F).var()
        return fm > BLUR_THRESHOLD
    except:
        return False

# Original alignment function with fixed syntax
def align_face(face_image, landmarks):
    try:
        left_eye = np.array(landmarks['left_eye'], dtype=np.float32)
        right_eye = np.array(landmarks['right_eye'], dtype=np.float32)

        # Validate eye coordinates
        if (np.any(np.isnan(left_eye))) or \
           (np.any(np.isnan(right_eye))) or \
           (np.any(left_eye < 0)) or \
           (np.any(right_eye < 0)) or \
           (left_eye[0] >= face_image.shape[1]) or \
           (left_eye[1] >= face_image.shape[0]) or \
           (right_eye[0] >= face_image.shape[1]) or \
           (right_eye[1] >= face_image.shape[0]):
            return face_image

        # Convert to integers for image operations
        left_eye = left_eye.astype(int)
        right_eye = right_eye.astype(int)

        # Calculate rotation angle
        dY = right_eye[1] - left_eye[1]
        dX = right_eye[0] - left_eye[0]
        angle = np.degrees(np.arctan2(dY, dX))

        # Calculate rotation center
        eyes_center = ((left_eye[0] + right_eye[0]) // 2,
                       (left_eye[1] + right_eye[1]) // 2)

        # Create rotation matrix
        M = cv2.getRotationMatrix2D(eyes_center, angle, scale=1.0)
        
        # Perform affine transformation
        aligned = cv2.warpAffine(
            face_image, 
            M, 
            (160, 160), 
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REPLICATE
        )
        
        return aligned

    except Exception as e:
        print(f"Alignment error: {str(e)}")
        return face_image

# Original routes with added functionality
@app.route('/')
def index():
    initialize_attendance()
    return render_template('index.html')

# Original registration endpoint with improvements
@app.route('/register_student', methods=['POST'])
def register_student():
    try:
        student_number = request.form['student_number']
        image_file = request.files['student_image']
        
        if not student_number or not image_file:
            return jsonify({"status": "error", "message": "Missing data"}), 400
        
        # Original folder structure
        student_dir = os.path.join(TRAIN_FOLDER, student_number)
        os.makedirs(student_dir, exist_ok=True)
        
        # Original image saving logic
        filename = f"{student_number}_{int(time.time())}.jpg"
        image_path = os.path.join(student_dir, filename)
        image_file.save(image_path)
        
        # Original label encoder update
        classes = list(label_encoder.classes_)
        if student_number not in classes:
            classes.append(student_number)
            label_encoder.classes_ = np.array(classes)
            np.save(LABEL_CLASSES_PATH, label_encoder.classes_)
        
        return jsonify({"status": "success", "message": "Student registered"})
    
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

# Original processing endpoint with exact parameter names
@app.route('/process_images', methods=['POST'])
def process_images():
    try:
        if 'images' not in request.files:
            return jsonify({"error": "No files uploaded"}), 400
        
        files = request.files.getlist('images')
        
        # Original face processing logic
        for file in files:
            in_memory_file = file.read()

            if not in_memory_file:
                continue  # Skip empty files

            image_np = np.frombuffer(in_memory_file, dtype=np.uint8)
            frame = cv2.imdecode(image_np, cv2.IMREAD_COLOR)

            if frame is None:
                continue  # Skip invalid/corrupted files

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) 
            
            # Original face detection and recognition flow
            faces = detector.detect_faces(rgb_frame)
            for face in faces:
                if face['confidence'] < 0.9:
                    continue
                
                x, y, w, h = face['box']
                face_region = rgb_frame[y:y+h, x:x+w]
                if face_region.size == 0:
                    continue
                
                # Original quality check
                if not is_face_quality_good(face_region.copy()):
                    continue
                
                # Original alignment and processing
                aligned_face = align_face(face_region, face['keypoints'])
                processed_face = cv2.resize(aligned_face, (160, 160))
                processed_face = cv2.cvtColor(processed_face, cv2.COLOR_RGB2GRAY)
                processed_face = cv2.equalizeHist(processed_face)
                processed_face = cv2.cvtColor(processed_face, cv2.COLOR_GRAY2RGB)
                processed_face = processed_face.astype("float32") / 255.0
                
                # Original prediction logic
                embedding = facenet_model.predict(np.expand_dims(processed_face, axis=0))[0]
                predictions = trained_model.predict(np.expand_dims(embedding, axis=0))[0]
                
                if np.max(predictions) > CONFIDENCE_THRESHOLD:
                    predicted_index = np.argmax(predictions)
                    label = label_encoder.classes_[predicted_index]
                    log_attendance(label)

        # Original attendance handling
        records = []
        for student, data in session['attendance'].items():
            records.append({
                'Student Number': student,
                'Status': data['Status'],
                'Time': data.get('Time', 'N/A')
            })
        
        df = pd.DataFrame(records)
        excel_stream = io.BytesIO()
        df.to_excel(excel_stream, index=False)
        excel_stream.seek(0)
        
        return send_file(
            excel_stream,
            mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            as_attachment=True,
            download_name=f'attendance_{datetime.now().strftime("%Y%m%d_%H%M%S")}.xlsx'
        )

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

# Maintained original session management
def log_attendance(student_number):
    current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    if 'attendance' not in session:
        session['attendance'] = {}

    session['attendance'][student_number] = {'Status': 'Present', 'Time': current_time}

    # Save attendance to Excel immediately
    records = []
    for student, data in session['attendance'].items():
        records.append({
            'Student Number': student,
            'Status': data['Status'],
            'Time': data.get('Time', 'N/A')
        })

    df = pd.DataFrame(records)
    df.to_excel(ATTENDANCE_FILE_PATH, index=False)





def initialize_attendance():
    all_students = label_encoder.classes_.tolist()
    session['attendance'] = {student: {'Status': 'Absent', 'Time': ''} 
                            for student in all_students}

# Rest of the original code remains unchanged
# ... [Keep the original video feed, camera controls, and download endpoints exactly as provided]

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

def generate_frames():
    global vs, face_history
    vs = VideoStream(src=0).start()
    attendance_logged = {}
    frame_counter = 0
    
    try:
        while True:
            frame = vs.read()
            if frame is None:
                continue
            
            frame_counter += 1
            if frame_counter % FRAME_SKIP != 0:
                continue

            frame = cv2.resize(frame, (800, 600))
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            try:
                faces = detector.detect_faces(rgb_frame)
            except:
                continue
            
            for face in faces:
                x, y, w, h = face['box']
                x, y = max(0, x), max(0, y)
                w = min(w, frame.shape[1] - x)
                h = min(h, frame.shape[0] - y)
                
                # Skip small or low-confidence detections
                if w < MIN_FACE_SIZE or h < MIN_FACE_SIZE:
                    continue
                if face['confidence'] < 0.90:
                    continue
                
                face_region = rgb_frame[y:y+h, x:x+w]
                if face_region.size == 0:
                    continue
                
                # Quality check with fallback
                if not is_face_quality_good(face_region.copy()):
                    cv2.putText(frame, "Low Quality", (x, y-30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
                    continue
                
                try:
                    # Alignment and preprocessing
                    aligned_face = align_face(face_region, face['keypoints'])
                    
                    # Ensure 3-channel RGB format
                    if aligned_face.ndim == 2:
                        aligned_face = cv2.cvtColor(aligned_face, cv2.COLOR_GRAY2RGB)
                    elif aligned_face.shape[2] == 4:
                        aligned_face = cv2.cvtColor(aligned_face, cv2.COLOR_RGBA2RGB)
                        
                    processed_face = cv2.resize(aligned_face, (160, 160))
                    processed_face = cv2.cvtColor(processed_face, cv2.COLOR_RGB2GRAY)
                    processed_face = cv2.equalizeHist(processed_face)
                    processed_face = cv2.cvtColor(processed_face, cv2.COLOR_GRAY2RGB)
                    processed_face = processed_face.astype("float32") / 255.0
                    
                    # Feature extraction
                    embedding = facenet_model.predict(np.expand_dims(processed_face, axis=0))[0]
                    predictions = trained_model.predict(np.expand_dims(embedding, axis=0))[0]
                    
                    # Prediction processing
                    sorted_probs = np.sort(predictions)[::-1]
                    confidence = sorted_probs[0]
                    
                    # Multi-stage verification
                    label = "Unknown"
                    if confidence > CONFIDENCE_THRESHOLD:
                        if (sorted_probs[0] - sorted_probs[1]) >= 0.3:
                            predicted_index = np.argmax(predictions)
                            try:
                                label = label_encoder.classes_[predicted_index]
                                # Temporal smoothing
                                if label not in face_history:
                                    face_history[label] = deque(maxlen=5)
                                face_history[label].append(confidence)
                                avg_confidence = np.mean(face_history[label])
                                if avg_confidence < CONFIDENCE_THRESHOLD:
                                    label = "Unknown"
                            except IndexError:
                                label = "Unknown"

                    # Attendance logging with cooldown
                    current_time = time.time()
                    if label != "Unknown":
                        last_logged = attendance_logged.get(label, 0)
                        if current_time - last_logged > ATTENDANCE_COOLDOWN:
                            log_attendance(label)
                            attendance_logged[label] = current_time

                    # Visual feedback
                    color = (0, 255, 0) if label != "Unknown" else (0, 0, 255)
                    cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                    text = f"{label} ({confidence:.2f})" if label != "Unknown" else "Unknown"
                    cv2.putText(frame, text, (x, y-10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                    
                    if label == "Unknown":
                        cv2.putText(frame, "UNVERIFIED", (x, y+h+20),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)

                except Exception as e:
                    print(f"Face processing warning: {str(e)}")
                    continue

            ret, buffer = cv2.imencode('.jpg', frame)
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
    
    finally:
        if vs:
            vs.stop()
        cv2.destroyAllWindows()

@app.route('/stop_camera', methods=['POST'])
def stop_camera():
    global vs
    if vs:
        vs.stop()
        vs = None
        return jsonify({'status': 'success', 'message': 'Camera stopped'})
    return jsonify({'status': 'error', 'message': 'No active camera'})

def initialize_attendance():
    all_students = label_encoder.classes_.tolist()
    session['attendance'] = {student: {'Status': 'Absent', 'Time': ''} 
                            for student in all_students}

@app.route('/download_attendance', methods=['GET'])
def download_attendance():
    records = []
    for student, data in session.get('attendance', {}).items():
        records.append({
            'Student Number': student,
            'Status': data['Status'],
            'Time': data.get('Time', 'N/A')
        })
    
    df = pd.DataFrame(records)
    excel_path = os.path.join(app.root_path, "attendance.xlsx")
    df.to_excel(excel_path, index=False)
    initialize_attendance()
    return send_file(excel_path, as_attachment=True)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, threaded=True, use_reloader=False)