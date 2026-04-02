from flask import Flask, render_template, Response, jsonify
import cv2
import numpy as np

app = Flask(__name__)

# Load the models
emotion_labels = ['Neutral', 'Happy', 'Surprise', 'Sad', 'Angry', 'Disgust', 'Fear', 'Contempt']
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
emotion_net = cv2.dnn.readNetFromONNX('emotion-ferplus-8.onnx')

camera_active = False
cap = None

# Global state to share with the frontend
latest_data = {
    'emotion': 'Neutral',
    'confidence': 0.0,
    'all_scores': {label: 0.0 for label in emotion_labels}
}

def start_camera():
    global cap, camera_active
    if cap is None or not cap.isOpened():
        cap = cv2.VideoCapture(0)
    camera_active = True

def gen_frames():
    global latest_data, camera_active
    
    while camera_active:
        if cap is None:
            break
        success, frame = cap.read()
        if not success:
            break
        
        # We will keep the video frame clean. 
        # All boxes and text animations will be drawn directly by the Browser!
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))
        
        largest_face = None
        max_area = 0
        
        for (x, y, w, h) in faces:
            area = w * h
            if area > max_area:
                largest_face = (x, y, w, h)
                max_area = area
            
        if largest_face:
            (x, y, w, h) = largest_face
            roi_gray = gray[y:y+h, x:x+w]
            try:
                roi_resized = cv2.resize(roi_gray, (64, 64))
                blob = cv2.dnn.blobFromImage(roi_resized, 1.0, (64, 64), (0, 0, 0), swapRB=False, crop=False)
                emotion_net.setInput(blob)
                preds = emotion_net.forward()[0]
                
                # Softmax to get probabilities
                exp_preds = np.exp(preds - np.max(preds))
                probs = exp_preds / exp_preds.sum()
                
                emotion_idx = np.argmax(probs)
                emotion_text = emotion_labels[emotion_idx]
                confidence = probs[emotion_idx]
                
                # We calculate percentages for the bar chart
                scores = {emotion_labels[i]: float(probs[i] * 100) for i in range(len(emotion_labels))}
                
                # Get the frame width and height to normalize coordinates for frontend
                frame_h, frame_w = frame.shape[:2]
                
                latest_data = {
                    'emotion': emotion_text,
                    'confidence': float(confidence * 100),
                    'all_scores': scores,
                    'face_box': {
                        'x': float(x / frame_w),
                        'y': float(y / frame_h),
                        'w': float(w / frame_w),
                        'h': float(h / frame_h)
                    }
                }
            except Exception as e:
                pass
        else:
            latest_data['face_box'] = None

        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    start_camera()
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/data')
def api_data():
    return jsonify(latest_data)

if __name__ == '__main__':
    print("\nStarting the Advanced Emotion Web Server!")
    print("Go to your browser and open: http://127.0.0.1:5000")
    print("To open on your phone or another device on the same Wi-Fi, go to: http://192.168.1.87:5000\n")
    app.run(host='0.0.0.0', debug=True, port=5000, threaded=True, use_reloader=False)
