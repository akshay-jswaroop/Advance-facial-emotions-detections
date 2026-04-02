import cv2
import numpy as np

# Emotion labels for ferplus model
emotion_labels = ['Neutral', 'Happy', 'Surprise', 'Sad', 'Angry', 'Disgust', 'Fear', 'Contempt']

print("Loading face detector and emotion model...")
# 1. Load the pre-trained face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# 2. Load the downloaded ONNX emotion recognition model
emotion_net = cv2.dnn.readNetFromONNX('emotion-ferplus-8.onnx')

print("Starting webcam...")
# 3. Open the webcam (0 is usually the default webcam)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

print("Webcam successfully opened. Press 'q' to quit.")

while True:
    # 4. Read the frame from the webcam
    ret, frame = cap.read()
    if not ret:
        break
        
    # Convert the frame to grayscale for face detection and emotion recognition
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # 5. Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))
    
    for (x, y, w, h) in faces:
        # Draw a rectangle around the face
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        # Extract the face region (Region of Interest - ROI)
        roi_gray = gray[y:y+h, x:x+w]
        
        try:
            # 6. Prepare the face for the emotion model
            # The ferplus model expects a 64x64 grayscale image
            roi_resized = cv2.resize(roi_gray, (64, 64))
            
            # Convert to a blob (a format the model understands)
            blob = cv2.dnn.blobFromImage(roi_resized, 1.0, (64, 64), (0, 0, 0), swapRB=False, crop=False)
            
            # Set the input and run the model
            emotion_net.setInput(blob)
            preds = emotion_net.forward()
            
            # Get the index of the highest probability
            emotion_idx = np.argmax(preds)
            emotion_text = emotion_labels[emotion_idx]
            
            # Get the confidence score
            confidence = np.max(preds)
            # You can uncomment the below line to see raw network output logic if needed:
            # print(preds)
            
            # 7. Write the predicted emotion text above the face rectangle
            text = f"{emotion_text}"
            cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            
        except Exception as e:
            print("Error processing face:", e)

    # 8. Display the resulting frame on the screen
    cv2.imshow('Facial Emotion Recognition', frame)
    
    # 9. Wait for a key press. If 'q' is pressed, break the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up: stop the webcam and close all viewing windows
cap.release()
cv2.destroyAllWindows()
