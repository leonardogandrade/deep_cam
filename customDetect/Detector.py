import cv2, time, os
from matplotlib.image import BboxImage
import numpy as np
import tensorflow as tf
import time
import json
from tensorflow.python.keras.utils.data_utils import get_file
from threading import Timer
np.random.seed(20)

class Detector:
    def __init__(self):
        self.fenceArea = []
        self.polyNodes = 4 # Number of nodes to draw a polygon
        self.totalTime = 0
        self.outputPath = os.path.join(os.getcwd(), "output")
        self.modelName = "ssd_mobilenet_v2_320x320_coco17_tpu-8"
        self.cacheDir = "./models"
        self.cx = []
        self.cy = []

    def readClasses(self, classesFilePath):
        with open(classesFilePath, 'r') as f:
            self.classesList = f.read().splitlines()
            # colors
            self.colorList = np.random.uniform(low=0, high=255, size=(len(self.classesList), 3))
            #print(len(self.classesList), len(self.colorList))
    
    def config(self):
        if not os.path.exists("output"):
            os.makedirs("output")

    def downloadModel(self, modelUrl):
        fileName = os.path.basename(modelUrl)
        
        self.modelName = fileName[:fileName.index('.')]
        self.cacheDir = "./models"
        os.makedirs(self.cacheDir, exist_ok=True)
        get_file(fname=fileName, origin=modelUrl, cache_dir=self.cacheDir, cache_subdir="checkfenceArea", extract=True)

    def loadModel(self):
        print("Loading Model " + self.modelName)
        tf.keras.backend.clear_session()
        self.model = tf.saved_model.load(os.path.join(self.cacheDir, "checkfenceArea", self.modelName, "saved_model"))

        print("Model " + self.modelName + " loaded successfully...")

    def checkInside(self, cx, cy):
        '''cx = center x of the detected object
           cy = center y of the detected object'''
        if(len(self.fenceArea) == self.polyNodes):
            if cv2.pointPolygonTest(np.array(self.fenceArea, np.int32), (int(cx), int(cy)), False) == 1.0:
                return True
            elif cv2.pointPolygonTest(np.array(self.fenceArea, np.int32), (int(cx), int(cy)), False) == -1.0:
                return False

    def output(self, payload, outputName):
        self.config()
        filename = os.path.join(self.outputPath, outputName + ".json")
        
        if os.path.exists(filename):
            os.remove(filename)

        file = open(filename, "w")
        json.dump(payload, file, indent=5)
        file.close()

    def getMovement(self, cx, cy):
        self.cx.append(cx)
        self.cy.append(cy)
        Timer(1, self.getMovement).start()

    def createBoundingBox(self, image, threshold= 0.5):
        inputTensor = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2RGB)
        inputTensor = tf.convert_to_tensor(inputTensor, dtype=tf.uint8)
        inputTensor = inputTensor[tf.newaxis,...]
        self.isInside = False
        detections = self.model(inputTensor)

        bboxs = detections['detection_boxes'][0].numpy()
        classIndexes = detections['detection_classes'][0].numpy().astype(np.int32)
        classScores = detections['detection_scores'][0].numpy()

        imH, imW, imC = image.shape
        bboxIdx = tf.image.non_max_suppression(bboxs, classScores, max_output_size=50, iou_threshold=threshold, score_threshold=threshold)

        print(bboxIdx)

        if len(bboxIdx) != 0:
            for i in bboxIdx:
                bbox = tuple(bboxs[i].tolist())
                classConfidence = round(100 * classScores[i])
                classIndex = classIndexes[i]
                
                # Only Person
                if classIndex in [1]:
                    classLabeltext = self.classesList[classIndex].upper()
                    classColor = self.colorList[classIndex]

                    displayText = '{}: {}%'.format(classLabeltext, classConfidence)

                    ymin, xmin, ymax, xmax = bbox

                    xmin, xmax, ymin, ymax = (xmin * imW, xmax * imW, ymin * imH, ymax * imH)
                    xmin, xmax, ymin, ymax = int(xmin), int(xmax), int(ymin), int(ymax)

                    # center point of a detected object
                    cx = int((xmin + xmax) / 2)
                    cy = int((ymin + ymax) / 2)

                    self.isInside = self.checkInside(cx, cy)

                    #print("Is object inside fence: " + str(isInside) + " " + str(cx) + " " + str(cy) + " " + str(self.fenceArea))
                    
                    cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color=classColor, thickness=1)
                    cv2.circle(image, (cx, cy), 5, classColor, thickness= -1)
                    
                    # get cx, cy data to save as output to charts
                    self.getMovement(cx / 100, cy / 100)
                    cv2.putText(image, displayText, (xmin, ymin - 10), cv2.FONT_HERSHEY_PLAIN, 1, classColor, 2)

                    ############################### Draw box corners ###############################
                    lineWidth = min(int((xmax - xmin) * 0.2), int((ymax - ymin) * 0.2))

                    cv2.line(image, (xmin, ymin), (xmin + lineWidth, ymin), classColor, thickness=5)
                    cv2.line(image, (xmin, ymin), (xmin, ymin + lineWidth), classColor, thickness=5)

                    cv2.line(image, (xmax, ymin), (xmax - lineWidth, ymin), classColor, thickness=5)
                    cv2.line(image, (xmax, ymin), (xmax, ymin + lineWidth), classColor, thickness=5)

                    #################################################################################
                    cv2.line(image, (xmin, ymax), (xmin + lineWidth, ymax), classColor, thickness=5)
                    cv2.line(image, (xmin, ymax), (xmin, ymax - lineWidth), classColor, thickness=5)

                    cv2.line(image, (xmax, ymax), (xmax - lineWidth, ymax), classColor, thickness=5)
                    cv2.line(image, (xmax, ymax), (xmax, ymax - lineWidth), classColor, thickness=5)

        return (image, self.isInside)

    def predictImage(self, imagePath, threshold= 0.5):
        img = cv2.imread(imagePath)

        (image, isInside) = self.createBoundingBox(img, threshold)

        cv2.imwrite(os.path.join("results", self.modelName + ".jpeg"), image)
        cv2.imshow("Result", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows

    def mouseClick(self, event, x, y, flags, params):
        
        if len(self.fenceArea) > self.polyNodes:
            #self.totalTimeFence = 0
            self.fenceArea.clear()

        if event == cv2.EVENT_LBUTTONDOWN:
            self.fenceArea.append((x,y))

    def predictVideo(self, videoPath, threshold = 0.5):
        cap = cv2.VideoCapture(videoPath)

        if(cap.isOpened() == False):
            print("Error openning file...")
            return

        (success, image) = cap.read()

        startTimeVideo = time.time()
        totalTimeVideo = 0
        startTimeFence = 0
        self.totalTimeFence = 0
        prevPosition = False

        while success:
            # currentTime = time.time()
            # fps = 1 / (currentTime - startTimeFence)
            # startTimeFence = currentTime

            (BboxImage, insideFence) = self.createBoundingBox(image, threshold)
            # cv2.putText(BboxImage, "FPS: " + str(int(fps)), (20,70), cv2.FONT_HERSHEY_PLAIN, 2, (0,255,0), 2)
            
            isInside = "" if insideFence == None else insideFence

            if insideFence and prevPosition is False:
                startTimeFence = time.time()
                prevPosition = True
            elif insideFence is False and prevPosition is True and self.totalTimeFence == 0:
                endTime = time.time()
                self.totalTimeFence =  endTime - startTimeFence
                prevPosition = False

            if self.totalTimeFence > 0:
                cv2.putText(BboxImage, "Permanencia: " + "{:.2f}".format(self.totalTimeFence) + "sec" , (20,130), cv2.FONT_HERSHEY_PLAIN, 2, (0,255,0), 2)

            cv2.putText(BboxImage, "Dentro: " + str(isInside) , (20,100), cv2.FONT_HERSHEY_PLAIN, 2, (0,255,0), 2)
            cv2.putText(BboxImage, "Dentro: " + str(isInside) , (20,100), cv2.FONT_HERSHEY_PLAIN, 2, (0,255,0), 2)
            cv2.setMouseCallback('Result', self.mouseClick)

            key = cv2.waitKey(1) & 0xFF
            
            if key == ord("q"):
                break
            
            # Draw and clear a polygon
            
            if len(self.fenceArea) == self.polyNodes:
                cv2.polylines(image, [np.array(self.fenceArea, np.int32)], True, color=(0, 0, 200), thickness=3)

            cv2.imshow('Result', BboxImage)

            (success, image) = cap.read()
        
            payload = {
                "video": {
                    "totalTime": "{:.2f}".format(time.time() - startTimeVideo)
                },
                "fences": [
                    {
                    "fenceId": "prateleira",
                    "totalTimeFence": "{:.2f}".format(self.totalTimeFence),
                    }
                ],
                "coord": [
                    {
                        "x": [self.cx],
                        "y": [self.cy]
                    }
                ]
            }       

        self.output(payload, "fence-output")
        cv2.destroyAllWindows()
    