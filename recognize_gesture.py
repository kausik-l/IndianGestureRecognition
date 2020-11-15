#Pickle in Python is primarily used in serializing and deserializing a Python object structure.
#In other words,It's the process of converting a Python object into a byte stream to store it in a file/database,
#maintain program state across sessions, or transport data over the network.
import cv2, pickle
import numpy as np
import os
import sqlite3
from keras.models import load_model

#Loading our stored model from the compressed h5 file.
model = load_model('cnn_model_keras.h5')


#img.shape gives out number of rows, columns and no. of channels(For RGB Image)
def get_image_size():
	img = cv2.imread('gestures/0/100.jpg', 0)
	return img.shape

image_rows , image_columns = get_image_size()

#Converting image pixels to numpy array.
def process_img(img):
	img = cv2.resize(img, (image_rows, image_columns))
	img = np.array(img, dtype=np.float32)
	#1st dimension is intact( 1 represents no. of images) and the next 2
	#dimensions are image pixels and last dimension represents number of channels.
	img = np.reshape(img, (1, image_rows, image_columns, 1))
	return img

#Predicting the class of image and its probability.
#We return only the maximum probability out of all.
def keras_predict(model, image):
	processed = process_img(image)
	prediction_prob = model.predict(processed)[0]
	prediction_class = list(prediction_prob).index(max(prediction_prob))
	return max(prediction_prob), prediction_class

#Acces the related text with the prediction class from our database
def get_text_from_db(prediction_class):
	conn = sqlite3.connect("gesture_db.db")
	#Our prediction class is matched with the gesture id in database.
	#If it matches, the gesture name is accessed.
	query = "SELECT g_name FROM gesture WHERE g_id="+str(prediction_class)
	cursor = conn.execute(query)
	for row in cursor:
		return row[0]

#Splitting the sentence that the model predicts from gesture.
def split_sentence(text, nb_words):

	list_words = text.split(" ")
	length = len(list_words)
	splitted_sentence = []
	start = 0
	end = nb_words
	while length > 0:
		part = ""
		for word in list_words[start:end]:
			#Separate each word
			part = part + " " + word
		splitted_sentence.append(part)
		#Increment the start and end index to get the next part.
		start += nb_words
		end += nb_words
		length -= nb_words
	return splitted_sentence

#To display the text on a separate empty frame.
def display_text(space,splitted_text):
	y = 200
	for text in splitted_text:
		cv2.putText(space, text, (4, y), cv2.FONT_HERSHEY_TRIPLEX, 2, (255, 255, 255))
		#To separate each word in the text.
		y += 50

#Access the hand histogram we captured.
def get_hand_hist():
	with open("hist", "rb") as f:
		hist = pickle.load(f)
	return hist

#our final function which recognizes the captured gesture.
def recognize():
	cam = cv2.VideoCapture(1)
	if cam.read()[0] == False:
		cam = cv2.VideoCapture(0)
	hist = get_hand_hist()
	x, y, w, h = 300, 100, 300, 300
	while True:
		text = ""
		img = cam.read()[1]
		#As the image will be opposite, we flip it.
		img = cv2.flip(img, 1)
		img = cv2.resize(img, (640, 480))
		imgCrop = img[y:y+h, x:x+w]
		imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
		#If we have a histogram of our skin color, we can use back projection
		#method to find skin colored parts in the image.
		dst = cv2.calcBackProject([imgHSV], [0, 1], hist, [0, 180, 0, 256], 1)
		#To create a structuring element in the shape of element.
		disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(10,10))
		#The structuring element is used as kernel
		#keep this kernel above a pixel, add all the pixels below this kernel,
		#take the average, and replace the central pixel with the new average value.
		cv2.filter2D(dst,-1,disc,dst)
		#Remove Gaussian noise.
		blur = cv2.GaussianBlur(dst, (11,11), 0)
		#Remove salt and pepper noise.
		blur = cv2.medianBlur(blur, 15)
		#First argument is the optimal threshold value which is not required,
		#2nd output is our threhsold image.
		thresh = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
		#Merging the transformed channels back to an image.
		thresh = cv2.merge((thresh,thresh,thresh))
		#Color to Grayscale.
		thresh = cv2.cvtColor(thresh, cv2.COLOR_BGR2GRAY)
		thresh = thresh[y:y+h, x:x+w]
		#findContours function has changed the order of the returned values in later values.
		(openCV_ver,_,__) = cv2.__version__.split(".")
		if openCV_ver=='3':
			contours = cv2.findContours(thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[1]
		elif openCV_ver=='4':
			contours = cv2.findContours(thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[0]
		if len(contours) > 0:
			contour = max(contours, key = cv2.contourArea)
			if cv2.contourArea(contour) > 10000:
				x1, y1, w1, h1 = cv2.boundingRect(contour)
				save_img = thresh[y1:y1+h1, x1:x1+w1]

				if w1 > h1:
					save_img = cv2.copyMakeBorder(save_img, int((w1-h1)/2) , int((w1-h1)/2) , 0, 0, cv2.BORDER_CONSTANT, (0, 0, 0))
				elif h1 > w1:
					save_img = cv2.copyMakeBorder(save_img, 0, 0, int((h1-w1)/2) , int((h1-w1)/2) , cv2.BORDER_CONSTANT, (0, 0, 0))

				prediction_prob, prediction_class = keras_predict(model, save_img)
				#More than 80% probability
				if prediction_prob*100 > 80:
					text = get_text_from_db(prediction_class)
					print(text)
		#Creating empty space to display the text from predicition.
		space = np.zeros((480, 640, 3), dtype=np.uint8)
		splitted_text = split_sentence(text, 2)
		display_text(space, splitted_text)
		cv2.rectangle(img, (x,y), (x+w, y+h), (0,255,0), 2)
		res = np.hstack((img, space))
		cv2.imshow("Recognizing gesture", res)
		cv2.imshow("thresh", thresh)
		if cv2.waitKey(1) == ord('q'):
			break

keras_predict(model, np.zeros((50, 50), dtype=np.uint8))
recognize()
