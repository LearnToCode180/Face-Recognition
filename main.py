import cv2
import os

#Initialize the classifier because OpenCV provides pre-trained models on Haar features and Cascade classifiers.
#These models are located in OpenCV installation.
cascPath=os.path.dirname(cv2.__file__)+"/data/haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascPath)


#Apply faceCascade on webcam frames:
video_capture = cv2.VideoCapture(0)

#This is how can we display the video on full screen
codec = 0x47504A4D  # MJPG
video_capture.set(cv2.CAP_PROP_FPS, 30.0)
video_capture.set(cv2.CAP_PROP_FOURCC, codec)
video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

while True:
    # Capture frame-by-frame
    ret, frames = video_capture.read()

    gray = cv2.cvtColor(frames, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frames, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow('I will recognize your face', frames)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

#Release the capture frames
video_capture.release()
cv2.destroyAllWindows()

