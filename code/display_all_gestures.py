import cv2, os, random
import numpy as np

def get_image_size():
	img = cv2.imread('gestures/0/100.jpg', 0)
	return img.shape



#list of g_id s of all the gestures we added.
gestures = os.listdir('gestures/')
gestures.sort(key = int)
start = 0
end = 5
image_rows, image_columns = get_image_size()

#If no. of gestures is 40, we will have 8 gestures in each row.
#Just for a better representation
if len(gestures)%5 != 0:
	rows = int(len(gestures)/5)+1
else:
	rows = int(len(gestures)/5)

full_img = None

#For each row,
for i in range(rows):
	col_img = None
	#In each column,
	for j in range(start, end):
		#Show a random image from each gesture directory.j is one of the first five g_ids.
		img_path = "gestures/%s/%d.jpg" % (j, random.randint(1, 1200))
		img = cv2.imread(img_path, 0)
		#If there is no image present, take empty space.
		if np.any(img == None):
			img = np.zeros((image_columns, image_rows), dtype = np.uint8)
		#The image will be displayed in the jth column of that ith row.
		if np.any(col_img == None):
			col_img = img
		#Images horizontally stacked together in one row.
		else:
			col_img = np.hstack((col_img, img))

	start += 5
	end += 5
	if np.any(full_img == None):
		full_img = col_img
	#Stacking vertically one row after another.
	else:
		full_img = np.vstack((full_img, col_img))


cv2.imshow("gestures", full_img)
cv2.imwrite('full_img.jpg', full_img)
cv2.waitKey(0)
