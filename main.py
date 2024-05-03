from deepface import DeepFace
from keras.models import load_model
import cv2  # Install opencv-python
import numpy as np
import time
from Adafruit_IO import MQTTClient

AIO_USERNAME = "thanhjash"
AIO_KEY = "aio_PYxs93yxnerpAK2viqn9bdzL6XHA"
# REMEMBER TO CREATE 2 FEEDS
# Face recognition
# Emotion Accuracy
client = MQTTClient(AIO_USERNAME, AIO_KEY)
client.connect()

# Load the face cascade classifier (already included in OpenCV)
face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Start capturing video from webcam
cap = cv2.VideoCapture(0)

last_sent_time = time.time()

while True:
    # Read frame from webcam
    ret, frame = cap.read()

    # Convert frame to grayscale for face detection
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_classifier.detectMultiScale(frame_gray)

    # Process each detected face
    for face in faces:
        # Extract face region
        x, y, w, h = face
        face_frame = frame[y:y+h, x:x+w]

        # Predict emotions on the face with enforce_detection=False
        emotions = DeepFace.analyze(face_frame, actions=['emotion'], enforce_detection=False)

        # Extract dominant emotion and its confidence score from the list
        dominant_emotion, score = max(emotions[0]['emotion'].items(), key=lambda x: x[1])

        # Draw rectangle around the face
        cv2.rectangle(frame, (x, y), (x+w, y+h), color=(255, 0, 0), thickness=2)

        # Display the predicted emotion and its confidence score
        text = f"Emotion: {dominant_emotion} ({score:.2f}%)"
        cv2.putText(frame, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # Display the frame with detected faces and emotions (after processing)
    cv2.imshow("Face Emotion Detection", frame)

    # Check if enough time has passed since the last data was sent
    if time.time() - last_sent_time >= 2:
        # Publish the detected emotion and its accuracy to MQTT topics
        client.publish("Face recognition", dominant_emotion)
        client.publish("Emotion Accuracy", f"{score:.2f}%")

        # Update the time for tracking
        last_sent_time = time.time()

    

    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
