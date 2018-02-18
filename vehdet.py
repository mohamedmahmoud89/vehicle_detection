import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
import glob
from skimage.feature import hog
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
import time
import pickle
from sklearn.cross_validation import train_test_split
from scipy.ndimage.measurements import label
from moviepy.editor import VideoFileClip

# to display images	
def show_image(img,desc=''):
	# Plot the examples
	fig = plt.figure()
	plt.subplot(121)
	plt.imshow(img)
	plt.title(desc)
	plt.show()
		
# a function to return some characteristics of the dataset 
def load_data():
	cars = []
	notcars = []

	images = glob.glob('training/vehicles/vehicles/GTI_Far/*.png')
	for image in images:
		cars.append(image)
	images = glob.glob('training/vehicles/vehicles/GTI_Left/*.png')
	for image in images:
		cars.append(image)
	images = glob.glob('training/vehicles/vehicles/GTI_MiddleClose/*.png')
	for image in images:
		cars.append(image)
	images = glob.glob('training/vehicles/vehicles/GTI_Right/*.png')
	for image in images:
		cars.append(image)
	images = glob.glob('training/vehicles/vehicles/KITTI_extracted/*.png')
	for image in images:
		cars.append(image)
	images = glob.glob('training/non-vehicles/non-vehicles/Extras/*.png')
	for image in images:
		notcars.append(image)
	images = glob.glob('training/non-vehicles/non-vehicles/GTI/*.png')
	for image in images:
		notcars.append(image)

	return cars,notcars

# to extract the hog features 	
def get_hog_features(img, orient, pix_per_cell, cell_per_block):
	features, hog_image = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
								  cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=False, 
								  visualise=True, feature_vector=False)
	return features, hog_image
		
# a function to compute binned color features  
def bin_spatial(img, size=(16, 16)):
	# Use cv2.resize().ravel() to create the feature vector
	features = cv2.resize(img, size).ravel() 
	# Return the feature vector
	return features

# a function to compute color histogram features 
def color_hist(img, nbins=32, bins_range=(0, 256)):
	# Compute the histogram of the color channels separately
	channel1_hist = np.histogram(img[:,:,0], bins=nbins, range=bins_range)
	channel2_hist = np.histogram(img[:,:,1], bins=nbins, range=bins_range)
	channel3_hist = np.histogram(img[:,:,2], bins=nbins, range=bins_range)
	# Concatenate the histograms into a single feature vector
	hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
	# Return the individual histograms, bin_centers and feature vector
	return hist_features

# a function to extract features from a list of images
# this function call bin_spatial() and color_hist()
def extract_features(imgs, cspace='RGB',hist_bins=32, hist_range=(0, 256)):
	# Create a list to append feature vectors to
	features = []
	orient = 12
	pix_per_cell = 8
	cell_per_block = 2
	# Iterate through the list of images
	for file in imgs:
		# Read in each one by one
		image = cv2.imread(file)  
		# Apply color_hist() also with a color space option now
		image = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
		
		hist_features = color_hist(image, nbins=hist_bins, bins_range=hist_range)
		spatial_features = bin_spatial(image, size=(32,32))
		#hog features
		hog_features = []
		for channel in range(image.shape[2]):
			hog_feat,img = get_hog_features(image[:,:,channel],orient,pix_per_cell,cell_per_block)
			hog_features.extend(hog_feat)
		hog_features = np.ravel(hog_features)
		# Append the new feature vector to the features list
		features.append(np.concatenate((spatial_features, hist_features, hog_features)))
	# Return list of feature vectors
	return features

# to train the classifier	
def train_svc(car_features,notcar_features):
	# Create an array stack of feature vectors
	X = np.vstack((car_features, notcar_features)).astype(np.float64)						
	# Fit a per-column scaler
	X_scaler = StandardScaler().fit(X)
	# Apply the scaler to X
	scaled_X = X_scaler.transform(X)
	
	# Define the labels vector
	y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))

	# Split up data into randomized training and test sets
	rand_state = np.random.randint(0, 100)
	X_train, X_test, y_train, y_test = train_test_split(
		scaled_X, y, test_size=0.2, random_state=rand_state)
	print('Feature vector length:', len(X_train[0]))
	# Use a linear SVC 
	svc = LinearSVC()
	# Check the training time for the SVC
	svc.fit(X_train, y_train)
	# Check the score of the SVC
	print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))
	# Check the prediction time for a single sample
	n_predict = 10
	print('My SVC predicts: ', svc.predict(X_test[0:n_predict]))
	print('For these',n_predict, 'labels: ', y_test[0:n_predict])
	
	return svc,X_scaler
	
# a function that takes an image,
# start and stop positions in both x and y, 
# window size (x and y dimensions),  
# and overlap fraction (for both x and y)
def slide_window(img, x_start_stop=[None, None], y_start_stop=[None, None], 
					xy_window=(64, 64), xy_overlap=(0.5, 0.5)):
	# If x and/or y start/stop positions not defined, set to image size
	if x_start_stop[0] == None:
		x_start_stop[0] = 0
	if x_start_stop[1] == None:
		x_start_stop[1] = img.shape[1]
	if y_start_stop[0] == None:
		y_start_stop[0] = 0
	if y_start_stop[1] == None:
		y_start_stop[1] = img.shape[0]
	# Compute the span of the region to be searched	
	xspan = x_start_stop[1] - x_start_stop[0]
	yspan = y_start_stop[1] - y_start_stop[0]
	#print(xspan)
	#print(yspan)
	# Compute the number of pixels per step in x/y
	nx_pix_per_step = np.int(xy_window[0]*(1 - xy_overlap[0]))
	ny_pix_per_step = np.int(xy_window[1]*(1 - xy_overlap[1]))
	# Compute the number of windows in x/y
	nx_buffer = np.int(xy_window[0]*(xy_overlap[0]))
	ny_buffer = np.int(xy_window[1]*(xy_overlap[1]))
	nx_windows = np.int((xspan-nx_buffer)/nx_pix_per_step) 
	ny_windows = np.int((yspan-ny_buffer)/ny_pix_per_step) 
	#print(nx_windows)
	#print(ny_windows)
	# Initialize a list to append window positions to
	window_list = []
	# Loop through finding x and y window positions
	# Note: you could vectorize this step, but in practice
	# you'll be considering windows one by one with your
	# classifier, so looping makes sense
	for ys in range(ny_windows):
		for xs in range(nx_windows):
			# Calculate window position
			startx = xs*nx_pix_per_step + x_start_stop[0]
			endx = startx + xy_window[0]
			starty = ys*ny_pix_per_step + y_start_stop[0]
			endy = starty + xy_window[1]
			# Append window position to list
			window_list.append(((startx, starty), (endx, endy)))
	# Return the list of windows
	return window_list

# Here is the draw_boxes function
def draw_boxes(img, bboxes, color=(0, 0, 255), thick=6):
	# Make a copy of the image
	imcopy = np.copy(img)
	# Iterate through the bounding boxes
	for bbox in bboxes:
		# Draw a rectangle given bbox coordinates
		cv2.rectangle(imcopy, bbox[0], bbox[1], color, thick)
	# Return the image copy with boxes drawn
	return imcopy

# to extract a single image features
def single_image_features(image):
	# Create a list to append feature vectors to
	features = []
	orient = 12
	pix_per_cell = 8
	cell_per_block = 2
	
	feature_image = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
	hist_features = color_hist(feature_image)

	spatial_features = bin_spatial(feature_image, size=(32,32))
	features.append(spatial_features)
	features.append(hist_features)
	hog_features = []
	for channel in range(feature_image.shape[2]):
		hog_feat,img = get_hog_features(feature_image[:,:,channel],orient,pix_per_cell,cell_per_block)
		hog_features.extend(hog_feat)
	hog_features = np.ravel(hog_features)
	# Append the new feature vector to the features list
	features.append(hog_features)
	# Return list of feature vectors
	return np.concatenate(features)
	#return (features)

# a function you will pass an image 
# and the list of windows to be searched (output of slide_windows())
def search_windows(img, windows, clf, scaler):

	#1) Create an empty list to receive positive detection windows
	on_windows = []
	#2) Iterate over all windows in the list
	for window in windows:
		#3) Extract the test window from original image
		test_img = cv2.resize(img[window[0][1]:window[1][1], window[0][0]:window[1][0]], (64, 64))	  
		#4) Extract features for that window using single_img_features()
		features = single_image_features(test_img)
		#5) Scale extracted features to be fed to classifier
		test_features = scaler.transform(np.array(features).reshape(1, -1))
		#6) Predict using your classifier
		prediction = clf.predict(test_features)
		#7) If positive (prediction == 1) then save the window
		if prediction == 1:
			on_windows.append(window)
	#8) Return windows for positive detections
	return on_windows
	
# to highlight the boxes with positive prediction
def add_heat(heatmap, bbox_list):
	# Iterate through list of bboxes
	for box in bbox_list:
		# Add += 1 for all pixels inside each bbox
		# Assuming each "box" takes the form ((x1, y1), (x2, y2))
		heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

	# Return updated heatmap
	return heatmap# Iterate through list of bboxes
	
# t apply the threshold
def apply_threshold(heatmap, threshold):
	# Zero out pixels below the threshold
	heatmap[heatmap <= threshold] = 0
	# Return thresholded map
	return heatmap

# to draw the boxes
def draw_labeled_bboxes(img, labels):
	# Iterate through all detected cars
	for car_number in range(1, labels[1]+1):
		# Find pixels with each car_number label value
		nonzero = (labels[0] == car_number).nonzero()
		# Identify x and y values of those pixels
		nonzeroy = np.array(nonzero[0])
		nonzerox = np.array(nonzero[1])
		# Define a bounding box based on min/max x and y
		bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
		# Draw the box on the image
		cv2.rectangle(img, bbox[0], bbox[1], (0,0,255), 6)
	# Return the image
	return img
	
# the final pipeline to be executed fr each photo in the video
def pipeline(image):
	global glb_labels
	global counter
	if counter == 0:
		window_1 = slide_window(image, x_start_stop=[896, 1280], y_start_stop=[400, 546], 
					xy_window=(128, 128), xy_overlap=(0.8, 0.7))
		window_2 = slide_window(image, x_start_stop=[768, 1056], y_start_stop=[400, 546], 
					xy_window=(128, 64), xy_overlap=(0.8, 0.7))
		windows = window_1 + window_2 

		hot_windows = search_windows(image, windows, svc, X_scalar)
		heat = np.zeros_like(image[:,:,0]).astype(np.float)
		# Add heat to each box in box list
		heat = add_heat(heat,hot_windows)

		# Apply threshold to help remove false positives
		heat = apply_threshold(heat,1)

		# Visualize the heatmap when displaying	
		heatmap = np.clip(heat, 0, 255)

		# Find final boxes from heatmap using label function
		glb_labels = label(heatmap)
		
	counter = (counter+1)%8
	draw_img = draw_labeled_bboxes(np.copy(image), glb_labels)
	
	return draw_imgs
#####################################
# the main routine

# load the data
cars,notcars = load_data()

# extraxt the features
car_features = extract_features(cars)
notcar_features = extract_features(notcars)

# train the classifier
svc,X_scalar = train_svc(car_features,notcar_features)

# reset the global vars
counter = 0
glb_labels = []

# the video processing
white_output = 'project_video_out.mp4'
clip1 = VideoFileClip("project_video.mp4")#.subclip(0,15)
white_clip = clip1.fl_image(pipeline) #NOTE: this function expects color images!!
white_clip.write_videofile(white_output, audio=False)
