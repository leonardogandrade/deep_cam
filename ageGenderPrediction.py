import numpy as np
import cv2
from keras.models import Model, Sequential
from keras.layers import Input, Convolution2D, ZeroPadding2D, MaxPooling2D, Flatten, Dropout, Activation
from keras.preprocessing.image import load_img, img_to_array
from keras.applications.imagenet_utils import preprocess_input
from keras.preprocessing import image
import os
from sys import argv

class AgeGender:
	def __init__(self):
		self.enableGenderIcons = True

		self.path = os.getcwd()
		self.imgDir = '/icons/'
		self.resDir = '/models/ageGender/'

		self.male_icon = cv2.imread(self.path + self.imgDir + "male_icon.jpeg")
		self.male_icon = cv2.resize(self.male_icon, (70, 70))

		self.female_icon = cv2.imread(self.path + self.imgDir + "female_icon.jpeg")
		self.female_icon = cv2.resize(self.female_icon, (70, 70))

		self.face_cascade = cv2.CascadeClassifier(self.path + "/config/" + 'haarcascade_frontalface_default.xml')

	def preprocess_image(self, image_path):
		img = load_img(image_path, target_size=(224, 224))
		img = img_to_array(img)
		img = np.expand_dims(img, axis=0)
		img = preprocess_input(img)

		return img

	def config(self):
		if not os.path.exists("results"):
			os.makedirs("results")

		if not os.path.exists("output"):
			os.makedirs("output")

		if not os.path.exists("images"):
			os.makedirs("images")
		

	def loadVggFaceModel(self):
		model = Sequential()
		model.add(ZeroPadding2D((1,1),input_shape=(224,224, 3)))
		model.add(Convolution2D(64, (3, 3), activation='relu'))
		model.add(ZeroPadding2D((1,1)))
		model.add(Convolution2D(64, (3, 3), activation='relu'))
		model.add(MaxPooling2D((2,2), strides=(2,2)))

		model.add(ZeroPadding2D((1,1)))
		model.add(Convolution2D(128, (3, 3), activation='relu'))
		model.add(ZeroPadding2D((1,1)))
		model.add(Convolution2D(128, (3, 3), activation='relu'))
		model.add(MaxPooling2D((2,2), strides=(2,2)))

		model.add(ZeroPadding2D((1,1)))
		model.add(Convolution2D(256, (3, 3), activation='relu'))
		model.add(ZeroPadding2D((1,1)))
		model.add(Convolution2D(256, (3, 3), activation='relu'))
		model.add(ZeroPadding2D((1,1)))
		model.add(Convolution2D(256, (3, 3), activation='relu'))
		model.add(MaxPooling2D((2,2), strides=(2,2)))

		model.add(ZeroPadding2D((1,1)))
		model.add(Convolution2D(512, (3, 3), activation='relu'))
		model.add(ZeroPadding2D((1,1)))
		model.add(Convolution2D(512, (3, 3), activation='relu'))
		model.add(ZeroPadding2D((1,1)))
		model.add(Convolution2D(512, (3, 3), activation='relu'))
		model.add(MaxPooling2D((2,2), strides=(2,2)))

		model.add(ZeroPadding2D((1,1)))
		model.add(Convolution2D(512, (3, 3), activation='relu'))
		model.add(ZeroPadding2D((1,1)))
		model.add(Convolution2D(512, (3, 3), activation='relu'))
		model.add(ZeroPadding2D((1,1)))
		model.add(Convolution2D(512, (3, 3), activation='relu'))
		model.add(MaxPooling2D((2,2), strides=(2,2)))

		model.add(Convolution2D(4096, (7, 7), activation='relu'))
		model.add(Dropout(0.5))
		model.add(Convolution2D(4096, (1, 1), activation='relu'))
		model.add(Dropout(0.5))
		model.add(Convolution2D(2622, (1, 1)))
		model.add(Flatten())
		model.add(Activation('softmax'))
		
		return model

	def ageModel(self):
		model = self.loadVggFaceModel()
		
		base_model_output = Sequential()
		base_model_output = Convolution2D(101, (1, 1), name='predictions')(model.layers[-4].output)
		base_model_output = Flatten()(base_model_output)
		base_model_output = Activation('softmax')(base_model_output)
		
		age_model = Model(inputs=model.input, outputs=base_model_output)
		age_model.load_weights(self.path + self.resDir + "age_model_weights.h5")
		
		return age_model

	def genderModel(self):
		model = self.loadVggFaceModel()
		
		base_model_output = Sequential()
		base_model_output = Convolution2D(2, (1, 1), name='predictions')(model.layers[-4].output)
		base_model_output = Flatten()(base_model_output)
		base_model_output = Activation('softmax')(base_model_output)

		gender_model = Model(inputs=model.input, outputs=base_model_output)
		gender_model.load_weights(self.path + self.resDir + "gender_model_weights.h5")
		
		return gender_model

	def predictImage(self, imgPath):
		age_model = self.ageModel()
		gender_model = self.genderModel()

		output_indexes = np.array([i for i in range(0, 101)])
		img =  cv2.imread(imgPath)
		gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		faces = self.face_cascade.detectMultiScale(img, 1.3, 5)

		
		for (x,y,w,h) in faces:
			if w > 130: #ignore small faces
				cv2.rectangle(img,(x,y),(x+w,y+h),(128,128,128),1)
				detected_face = img[int(y):int(y+h), int(x):int(x+w)]

				try:
					margin = 30
					margin_x = int((w * margin)/100); margin_y = int((h * margin)/100)
					detected_face = img[int(y-margin_y):int(y+h+margin_y), int(x-margin_x):int(x+w+margin_x)]
				except:
					print("detected face has no margin")

				try:
					#vgg-face expects inputs (224, 224, 3)
					detected_face = cv2.resize(detected_face, (224, 224))
					
					img_pixels = image.img_to_array(detected_face)
					img_pixels = np.expand_dims(img_pixels, axis = 0)
					img_pixels /= 255
					
					age_distributions = age_model.predict(img_pixels)
					apparent_age = str(int(np.floor(np.sum(age_distributions * output_indexes, axis = 1))[0]))
					
					gender_distribution = gender_model.predict(img_pixels)[0]
					gender_index = np.argmax(gender_distribution)
					
					if gender_index == 0: 
						gender = "F"
					else: gender = "M"
				
					#background for age gender declaration
					info_box_color = (46,200,255)
					
					cv2.rectangle(img,(x,y-70),(x+int(w/2)+ 30,y),info_box_color,cv2.FILLED)

					# if self.enableGenderIcons:
					if gender == 'M': 
						gender_icon = self.male_icon
					else: 
						gender_icon = self.female_icon

					cv2.putText(img, apparent_age, (x + 80, y - 15), cv2.FONT_HERSHEY_PLAIN, 4, (0, 111, 254), 3)
					img[y-70 : y-70 + self.male_icon.shape[0], x : x + self.male_icon.shape[1]] = gender_icon

				except Exception as e:
					print("exception",str(e))

		#cv2.imshow('img',img)
		cv2.imwrite(os.path.join(os.getcwd(), "results", "ageGender.png"), img)

	cv2.destroyAllWindows()


imgPath = argv[1]
predictor = AgeGender()
predictor.config()
predictor.predictImage(imgPath)