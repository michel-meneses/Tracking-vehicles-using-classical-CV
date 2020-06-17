from __future__ import division
import cv2
import numpy as np
from math import *

#--------------------------------------------
#		  AUXILIARY BLOCK FUNCTIONS
#--------------------------------------------

# return closed image
def closing(bny, dim):
	return erosion(dilation(bny, dim), dim);

# return dilation of a binary image
def dilation(bny, dim):
	
	# create mask matrix
	mask = 255*np.ones((dim, dim));
	
	# initialize output image
	out = np.zeros((bny.shape[0], bny.shape[1]));
	
	# find activated pixels positions
	actBny = np.where(bny > 0);
	
	# find activated mask pixels positions
	actMask = np.where(mask > 0);
	
	# apply mask over activated points
	for i, (x0,y0) in enumerate(zip(actBny[0], actBny[1])):
		for j, (x1,y1) in enumerate(zip(actMask[0], actMask[1])):
			idX = x0+x1;
			idY = y0+y1;
			if idX >= 0 and idX < out.shape[0] and idY >= 0 and idY < out.shape[1]:
				out[idX][idY] = 255;
	return out;

# return erosion of a binary image
def erosion(bny, dim):
	
	# create mask matrix
	mask = 255*np.ones((dim, dim));
	
	# initialize output image
	out = np.zeros((bny.shape[0], bny.shape[1]));
	
	# find activated pixels positions
	actBny = np.where(bny > 0);
	
	# apply mask over activated points
	for aux, i in enumerate(zip(actBny[0], actBny[1])):
		noMatch = False;
		for (k,aux) in np.ndenumerate(mask):
			if i[0]+k[0] < 0 or i[0]+k[0] >= out.shape[0] or i[1]+k[1] < 0 or i[1]+k[1] >= out.shape[1] or mask[k[0]][k[1]] != bny[i[0]+k[0]][i[1]+k[1]]:
				noMatch = True;
				break;
		
		if noMatch == False:
			out[i[0]][i[1]] = 255;
	return out;

# return list of features (area, position[x,y], box[x0,y0,x1,y1]) of segmented image
def getFeatures(segm):
	features = [];
	for i in range(1, int(np.max(segm) + 1)):
		index = np.where(segm == i);
		iX = np.sum(index[0]);
		iY = np.sum(index[1]);
		area = [np.sum(segm == i)];
		pos = [int(iX/area), int(iY/area)];
		box = [np.min(index[0]), np.min(index[1]), np.max(index[0]), np.max(index[1])];
		
		# add features
		features.append(area + pos + box);
	return np.asarray(features);
	
# return histogram matrix of an image
def getHistogram(img):
	hist = [];
	for i in range(256):
		hist.append( len(np.where(img[:,:]==i)[0]) / (img.shape[0]*img.shape[1]) );
	return np.asarray(hist);
	
# calculate Otsu's threshold for a given image
def getOtsuThreshold(img):
	
	# get image's histogram
	hist = getHistogram(img);
	
	#calculate global average pixels intensity
	gAv = np.sum(img)/(img.shape[0]*img.shape[1]);
	
	# calculate variances related to each possible threshold value
	variances = [];
	for tresh in range(256):
		p = np.sum(hist[0:tresh]);
		m = hist[0];
		for i in range(1,tresh+1):
			m = (m + i*hist[i])/2;
			
		# calculate variance
		if p != 1 and p != 0:
			variances.append(pow((gAv*p - m),2)/(p*(1-p)));
		else:
			variances.append(0);
	
	# return the best threshold
	return np.argmax(np.asarray(variances));

# return resized image based in a scale factor
def resize(img, scale):

	# initialize output image
	out = np.zeros((int(scale*img.shape[0]), int(scale*img.shape[1])));
	
	# initialize transformation matrixes
	T = np.matrix([[1,0,0], [0,1,0], [0,0,1]]);
	S = np.matrix([[1/scale,0,0], [0,1/scale,0], [0,0,1]]);
	M = np.linalg.inv(T)*S*T;
	M = M.astype(int);
	
	# apply transformation
	for i in np.nditer(np.arange(out.shape[0])):
		for j in np.nditer(np.arange(out.shape[1])):
			
			# calculate new coordinate
			newP = (M*np.matrix([[i],[j],[1]])).astype(int);
			
			# try to transform image
			if newP[0] >= 0 and newP[0] < img.shape[0] and newP[1] >= 0 and newP[1] < img.shape[1]:
				out[i,j] = img[newP[0], newP[1]];
	
	return out;

# return images subtraction
def subtract(img1, img2):
	sub = (img2-img1);
	sub[sub < 0] = 0;
	sub[sub > 255] = 255;
	return sub;

# return matrix of segmented regions of a binary image
def segment(bny):
	
	# initialize output image
	out = np.zeros((bny.shape[0], bny.shape[1]));
	
	# find activated pixels positions
	actBny = np.where(bny > 0);
	
	# initialize regions counter
	regions = 0;
	
	# segment
	for aux, (i,j) in enumerate(zip(actBny[0], actBny[1])):
		if i - 1 >= 0 and bny[i-1,j] > 0 and out[i-1,j] > 0:
			out[i,j] = out[i-1, j];
		elif j - 1 >= 0 and bny[i,j-1] > 0 and out[i,j-1] > 0:
			out[i,j] = out[i, j-1];
		elif i + 1 < bny.shape[0] and bny[i+1,j] > 0 and out[i+1,j] > 0:
			out[i,j] = out[i+1, j];
		elif j + 1 < bny.shape[1] and bny[i,j+1] > 0 and out[i,j+1] > 0:
			out[i,j] = out[i, j+1];
		else:
			regions = regions + 1;
			out[i,j] = regions;
	
	# remove duplicated regions
	for aux, (i,j) in enumerate(zip(actBny[0], actBny[1])):
		if i + 1 < bny.shape[0] and bny[i+1,j] > 0 and out[i,j] != out[i+1,j] > 0:
			out[ out == out[i,j] ] = out[i+1,j];
		if j + 1 < bny.shape[1] and bny[i,j+1] > 0 and out[i,j] != out[i,j+1] > 0:
			out[ out == out[i,j] ] = out[i,j+1];
	
	# rescale output
	regions = np.unique(out);
	for r in np.nditer(regions):
		np.place(out, out == r, np.where(regions == r)[0]);
	return out
	
# convert image to binary
def toBinary(img, threshold):

	# define threshold
	threshold = threshold#getOtsuThreshold(img);
	
	# return binary image
	img[img < threshold] = 0;
	img[img >= threshold] = 255;
	return img;

# convert image to grayscale
def toGrayscale(img):
	return img[:,:,0]/3 + img[:,:,1]/3 + img[:,:,2]/3;

#--------------------------------------------
#			 CONTROL FUNCTIONS
#--------------------------------------------

# given two frames return the follow car features array: [area,pos[x,y],box[x0,y0,x1,y1]
def extractFeatures(prevImg, nextImg):

	# convert frames to grayscale
	prevImg = toGrayscale(prevImg);
	nextImg = toGrayscale(nextImg);

	# subtract both frames in order to do a segmentation
	img = subtract(prevImg, nextImg);
	
	# convert image to binary
	img = toBinary(img, 20);
	
	# close image
	img = closing(img, 13);
	
	# segment image
	img = segment(img);
	
	# extract features
	features = getFeatures(img);
	
	# filter regions by theur areas
	finalFeatures = [];
	minArea = 60;
	maxArea = 1500;
	for f in features:
		if f[0] >= minArea and f[0] <= maxArea and f[1] > img.shape[0]/2 and f[1] < img.shape[0]*0.9 and (f[2] > img.shape[1]*0.55 or f[2] < img.shape[1]*0.4):
			finalFeatures.append(f);
	return np.asarray(finalFeatures);

# calculate velocity based on two arrays of features
def calculateVelocity(prevFeatures, nextFeatures, ellapsedTime):
	vel = [];
	prevF = prevFeatures.tolist();
	nextF = nextFeatures.tolist();
	for f1 in nextF:

		# find previous feature related to the current one
		mDist = 0;
		pF = [];
		for f0 in prevF:
			dist = sqrt((f1[1]-f0[1])**2 + (f1[2]-f0[2])**2);
			if mDist == 0 or dist < mDist:
				mDist = dist;
				pF = f0;
		
		# add velocity and remove feature from previous features array
		if pF != []:
			vel.append([pF,f1,mDist/ellapsedTime]);
			prevF.remove(pF);
	return np.asarray(vel);
	
# draw results over segmented image
def drawResults(prevImg, nextImg, prevFeatures, nextFeatures, ellapsedTime):
	
	# initialize output
	out = nextImg.copy();
	
	# calculate velocities to be shown
	vel = calculateVelocity(prevFeatures, nextFeatures, ellapsedTime);
	
	for v in vel:
	
		# draw boxes above regions
		cv2.circle(out, (v[1][2], v[1][1]), 1, 200, -1);
		cv2.rectangle(out, (v[1][4], v[1][3]), (v[1][6], v[1][5]), 127, 1);
		font = cv2.FONT_HERSHEY_SIMPLEX;
		cv2.putText(out, str(round(v[2],1)) + ' px/s', (v[1][4], v[1][3] - 5), font, 0.3, (0,255,0), 1, cv2.LINE_AA);
	return out;
	
#--------------------------------------------
#				MAIN CODE
#--------------------------------------------
	
# load video to be processed	
video = cv2.VideoCapture('video.mp4');

# get some proprieties from the video
fps = int(video.get(cv2.CAP_PROP_FPS));
w = int(video.get(cv2.CAP_PROP_FRAME_WIDTH));
h = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT));
frames = video.get(cv2.CAP_PROP_FRAME_COUNT);

# set compression format of the output video
fourcc = cv2.VideoWriter_fourcc(*'H264')

# initialize output video object
out = cv2.VideoWriter('output.avi', fourcc, fps, (w, h), True);

# catch initial frame
prevImg = None
while video.isOpened() and prevImg is None:
	ret, frame = video.read()
	if ret == True:
		prevImg = frame

# initialize previous and next segmentation features
prevFeatures = None;
nextFeatures = None;

# initialize current frame counter
crtFrm = 1;

# define amount of frames skipped between each segmentation
skipFrm = 4;

# start video processing
while video.isOpened() and crtFrm < 0.02*frames:

	# load image
	ret, nextImg = video.read();
	if ret==True:
		
		# print progress
		print ('Processing... ({0:.2f}%)'.format(100*crtFrm/frames));
		
		# resize frames
		scale = 1/4;
		resPrevImg = cv2.resize(prevImg, (int(prevImg.shape[1]*scale), int(prevImg.shape[0]*scale)), interpolation = cv2.INTER_AREA);
		resNextImg = cv2.resize(nextImg, (int(nextImg.shape[1]*scale), int(nextImg.shape[0]*scale)), interpolation = cv2.INTER_AREA);
		
		# calculate velocity after skip some frames
		if crtFrm%skipFrm == 0:
			
			# segment and calculate features of segmented regions of frames
			aux = extractFeatures(resPrevImg, resNextImg);
			if prevFeatures is None:
				prevFeatures = aux;
			else:
				nextFeatures = aux;
		
		# update previous frame
		prevImg = nextImg;
		
		# update frames counter
		crtFrm = crtFrm + 1;
		
		if nextFeatures is not None:
		
			# draw result image
			result = drawResults(resPrevImg, resNextImg, prevFeatures, nextFeatures, skipFrm/fps);
			
			# resize result
			scale = 1/scale;
			result = cv2.resize(result, (int(result.shape[1]*scale), int(result.shape[0]*scale)), interpolation = cv2.INTER_AREA);
			
			# write frame in output video
			out.write(result);
			
			# update previous features
			prevFeatures = nextFeatures;

# close video files
video.release()
video.release()
out.release()