import main
import cv2
video = cv2.VideoCapture(0)

test = "Unknown"
while True:
    if(test == "Unknown"):
        while True:
            rep, frame = video.read()
            test = main.face_detector(frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
    elif(test != "Unknown"):
        print(test)