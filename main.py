import cv2
import numpy as np
from deepface import DeepFace
import tensorflow as tf
import os

# Verify that TensorFlow is using the GPU
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    try:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
        tf.config.experimental.set_virtual_device_configuration(
            physical_devices[0],
            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4096)]
        )
        # print(f"TensorFlow GPU setup: {physical_devices[0]}")
    except RuntimeError as e:
        print(e)
# else:
    # print("No GPU detected. Running on CPU.")

# Load pre-trained face detection model from OpenCV
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Initialize webcam
cap = cv2.VideoCapture(0)

# Define the dictionary for reference images and their labels
reference_images = {
    "rawan": ["images/2023_12_19_13_39_IMG_5736.JPG", "images/426851309_7552795658064400_7666217970037880622_n.jpg"],
    "omar": ["images/379470256_3632000777031852_888413986015996647_n.jpg"]
}

# Pre-process reference images and find embeddings
reference_embeddings = {}
for label, image_paths in reference_images.items():
    embeddings = []
    for image_path in image_paths:
        embedding = DeepFace.represent(img_path=image_path, model_name='VGG-Face')[0]["embedding"]
        embeddings.append(embedding)
    reference_embeddings[label] = embeddings

def recognize_face(face_img):
    try:
        # Save the face_img to a temporary file for DeepFace to process
        temp_face_path = "temp_face.jpg"
        cv2.imwrite(temp_face_path, face_img)
        
        # Find the embedding for the current face
        current_embedding = DeepFace.represent(img_path=temp_face_path, model_name='VGG-Face')[0]["embedding"]
        
        # Compare with reference embeddings
        min_distance = float("inf")
        label = "Unknown"
        for ref_label, ref_embeddings in reference_embeddings.items():
            for ref_embedding in ref_embeddings:
                distance = np.linalg.norm(np.array(ref_embedding) - np.array(current_embedding))
                if distance < min_distance:
                    min_distance = distance
                    label = ref_label
        
        return label
    except Exception as e:
        # print("Error in recognizing face:", e)
        return "Unknown"

def face_detected():
    while True:
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        
        for (x, y, w, h) in faces:
            face_img = frame[y:y+h, x:x+w]
            label = recognize_face(face_img)
            
            # Draw rectangle around the face
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            
            # Put label text above the face rectangle
            cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)

        # Display the resulting frame
        cv2.imshow('Face Recognition', frame)

        # Press 'q' to exit the video stream
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
    # Release the webcam and close the window
    cap.release()
    cv2.destroyAllWindows()


def face_detector(frame):
    while True:
        frame = frame
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        
        for (x, y, w, h) in faces:
            face_img = frame[y:y+h, x:x+w]
            label = recognize_face(face_img)
            
            # Draw rectangle around the face
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            
            # Put label text above the face rectangle
            cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
            return label
        # Display the resulting frame
        cv2.imshow('Face Recognition', frame)

        # Press 'q' to exit the video stream
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
    # Release the webcam and close the window
    cap.release()
    cv2.destroyAllWindows()
if __name__ == '__main__':
    face_detected()
