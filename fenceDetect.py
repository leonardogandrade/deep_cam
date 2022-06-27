from sys import argv
from customDetect.Detector import *

videoPath = argv[1]
imagePath = argv[1]

threshold = 0.5
classFile = "config/coco.names"
detector = Detector()
detector.readClasses(classFile)

detector.loadModel()
detector.predictVideo(videoPath, threshold)