import cv2
import numpy as np

FACE_DETECTION_PATH = '/models/haarcascade_frontalface_default.xml'


def video_predict(model, emotions, pwd):
    face_haar_cascade = cv2.CascadeClassifier(pwd + FACE_DETECTION_PATH)

    cap = cv2.VideoCapture(4) # set the available camera here!

    # Capture settings
    cap.set(cv2.CAP_PROP_BRIGHTNESS, 30)
    cap.set(cv2.CAP_PROP_CONTRAST, 50)
    cap.set(cv2.CAP_PROP_SATURATION, 60)

    while(True):
        # capture frame
        ret, frame = cap.read()
        
        # change frame color to gray
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # face detection with OpenCV haar cascade
        faces_detected = face_haar_cascade.detectMultiScale(gray)
        for (x,y,w,h) in faces_detected:

            # draw a rectangle around face
            cv2.rectangle(gray,(x,y), (x+w,y+h), (255,0,0))

            # crop the face
            face = gray[y:y+w,x:x+h]

            # resize image to fit model's input
            face = cv2.resize(face,(48,48))

            # add 1 dimension to work as a tensor
            image_pixels = np.expand_dims(face, axis = 0)

        # predict emotion
        predictions = model.predict(image_pixels) # returns probabilities for each class

        max_index = np.argmax(predictions[0]) # 0 or 1  or 2 or ...  or 6

        # emotion_detection = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')
        emotion_prediction = emotions[max_index]

        cv2.putText(frame, emotion_prediction, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX,2,(0,255,0),3)  

        resized_image = cv2.resize(frame, (1000, 700))
        cv2.imshow('Emotion',resized_image)
        
        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()