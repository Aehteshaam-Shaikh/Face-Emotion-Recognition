from keras.models import model_from_json

import cv2
import numpy as np

model_json_file = 'model.json'
model_weights_file = 'model.h5'
with open(model_json_file, "r") as json_file:
    loaded_model_json = json_file.read()
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights(model_weights_file)


face_haar_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')  
emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

def image_predict(image):
    return emotions[np.argmax(loaded_model.predict(image))]

class VideoCamera:
    def __init__(self):
        self.video = cv2.VideoCapture(0)
    
    def __del__(self):
        self.video.release()

    def get_frame(self):
        ret,test_img=self.video.read()# captures frame and returns boolean value and captured image  
          
        
        gray_img= cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)  

        try:
            faces_detected = face_haar_cascade.detectMultiScale(gray_img, 1.3, 5) 

            for (x,y,w,h) in faces_detected:  
                cv2.rectangle(test_img,(x,y),(x+w,y+h),(0,0,255),thickness=3)  
                roi_gray=gray_img[y:y+h,x:x+w]#cropping region of interest i.e. face area from  image  
                roi_gray=cv2.resize(roi_gray,(48,48))  
                img = roi_gray.reshape((1,48,48,1))
                img = img /255.0

                max_index = np.argmax(loaded_model.predict(roi_gray[np.newaxis, :, :, np.newaxis]))

                  
                predicted_emotion = emotions[max_index]  

                cv2.putText(test_img, predicted_emotion, (int(x), int(y-5)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
        except:
            pass

        resized_img = cv2.resize(test_img, (1000, 600))

        _, jpeg = cv2.imencode('.jpg', resized_img)

        return jpeg.tobytes()