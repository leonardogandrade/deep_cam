import cv2
import numpy as np
import cv2
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.preprocessing.image import img_to_array, load_img
import json, os
from sys import argv
filePath = argv[1]
np.random.seed(20)

class Expressions:
    def __init__(self):
        self.modelPath = "models/expressions/"
        self.outputPath = os.path.join(os.getcwd(), "output")

    def readClases(self, filePath):
        with open(filePath, 'r') as file:
            self.classesList = file.read().splitlines()
            self.colorList = np.random.uniform(low=0, high=255, size=(len(self.classesList), 3))

    def loadModel(self, modelName):
        self.modelName = modelName

        model = Sequential()
        model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48,48,1)))
        model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Flatten())
        model.add(Dense(1024, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(7, activation='softmax'))

        model.load_weights(str(self.modelPath + self.modelName))
        
        self.model = model

        print("Model was load successfully...")

    def config(self):
        if not os.path.exists("results"):
            os.makedirs("results")

        if not os.path.exists("output"):
            os.makedirs("output")

        if not os.path.exists("images"):
            os.makedirs("images")

    def output(self, payload, outputName):
        filename = os.path.join(self.outputPath, outputName + ".json")
        
        if os.path.exists(filename):
            os.remove(filename)

        file = open(filename, "w")
        json.dump(payload, file, indent=5)
        file.close()

    def predictImage(self, frame):
        img = load_img(frame, target_size=(224,224))
        imgArray = img_to_array(img) / 255
        inputArray = np.array([imgArray])

        return np.argmax(self.model.predict(inputArray))

    def predictVideo(self, filePath=0):
        cap = cv2.VideoCapture(filePath)

        payload = {
            "expressions": {
                "Raiva": 0,
                "Aversao": 0,
                "Medo": 0,
                "Contente": 0,
                "Neutro": 0,
                "Triste": 0,
                "Surpreso": 0
            }
        }

        while True:
            ret, frame = cap.read()

            if not ret:
                break

            faceCascade = cv2.CascadeClassifier("config/haarcascade_frontalface_default.xml")
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = faceCascade.detectMultiScale(gray,scaleFactor=1.3, minNeighbors=5)
            
            for (x, y, w, h) in faces:
                
                roiGray = gray[y:y + h, x:x + w]
                croppedImg = np.expand_dims(np.expand_dims(cv2.resize(roiGray, (48, 48)), -1), 0)

                prediction = self.model.predict(croppedImg)
                maxIndex = np.argmax(prediction)

                if maxIndex in (0, 1, 2, 4, 5):
                    maxIndex = 4

                classColor = self.colorList[maxIndex]
                cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), classColor, 2)

                cv2.putText(frame, self.classesList[maxIndex],(x, y - 60), cv2.FONT_HERSHEY_SIMPLEX, 1, classColor, thickness=3)

                payload['expressions'][self.classesList[maxIndex]] += 1

            #cv2.imshow('Face expressions', cv2.resize(frame, (1600, 960), interpolation= cv2.INTER_CUBIC))
            cv2.imshow('Face expressions', frame)


            key = cv2.waitKey(10)
            if key == ord("q") & 0xFF:
                break
        
        self.output(payload, "expressions-output")    
        cap.release()
        cv2.destroyAllWindows()


expressions = Expressions()
expressions.config()
modelName = "model.h5"
classPath = "config/expressions.names"

expressions.loadModel(modelName)
expressions.readClases(classPath)

expressions.predictVideo(filePath)