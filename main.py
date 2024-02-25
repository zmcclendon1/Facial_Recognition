# imports
import cv2


#Models
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')

#Defining Faces and Smiles
def detect_faces_and_smiles(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    #Boxes and Colors
    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]

        smiles = smile_cascade.detectMultiScale(roi_gray, scaleFactor=1.8, minNeighbors=20)

        color = (255, 0, 0)  
        for (sx, sy, sw, sh) in smiles:
            color = (0, 255, 0)  
            cv2.rectangle(roi_color, (sx, sy), (sx+sw, sy+sh), color, 2)

        
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)

    return frame
#Handling Video Capture and UI Window
video_capture = cv2.VideoCapture(0)

if not video_capture.isOpened():
    print("Error: Could not open webcam")
    exit()

while True:
    ret, frame = video_capture.read()

    if not ret:
        print("Error: Failed to capture frame from webcam")
        break

    frame_with_boxes = detect_faces_and_smiles(frame)

    cv2.imshow("Face and Smile Detection", frame_with_boxes)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()



