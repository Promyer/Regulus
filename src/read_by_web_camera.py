#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import dlib
import imutils
from imutils import face_utils
from collections import OrderedDict
import argparse
from imutils.face_utils import FaceAligner
from imutils.face_utils import rect_to_bb
import cv2
import torch


# In[ ]:


def shape_to_np(shape, dtype="int"):
	# initialize the list of (x, y)-coordinates
	coords = np.zeros((68, 2), dtype=dtype)
 
	# loop over the 68 facial landmarks and convert them
	# to a 2-tuple of (x, y)-coordinates
	for i in range(0, 68):
		coords[i] = (shape.part(i).x, shape.part(i).y)
 
	# return the list of (x, y)-coordinates
	return coords
FACIAL_LANDMARKS_IDXS = OrderedDict([
	("mouth", (48, 68)),
	("right_eyebrow", (17, 22)),
	("left_eyebrow", (22, 27)),
	("right_eye", (36, 42)),
	("left_eye", (42, 48)),
	("nose", (27, 35)),
	("jaw", (0, 17))
])
class FaceAligner:
    def __init__(self, desiredLeftEye=(0.35, 0.35),
        desiredFaceWidth=256, desiredFaceHeight=None):
        self.FACIAL_LANDMARKS_IDXS = OrderedDict([
            ("mouth", (48, 68)),
            ("right_eyebrow", (17, 22)),
            ("left_eyebrow", (22, 27)),
            ("right_eye", (36, 42)),
            ("left_eye", (42, 48)),
            ("nose", (27, 35)),
            ("jaw", (0, 17))
        ])        
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor('../notebooks/models/shape_predictor_68_face_landmarks.dat')
        self.desiredLeftEye = desiredLeftEye
        self.desiredFaceWidth = desiredFaceWidth
        self.desiredFaceHeight = desiredFaceHeight
        if self.desiredFaceHeight is None:
            self.desiredFaceHeight = self.desiredFaceWidth
    def align(self, image):
        image = imutils.resize(image, width=800)              
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        #if (face_loc == None):       
        rects = self.detector(gray, 1)       
        #else:
        #    rects = [dlib.rectangle(face_loc[0], face_loc[3], face_loc[1], face_loc[2])]
        # loop over the face detections
        if (len(rects) == 0):
            return None
        rect = rects[0]
        try:
            # extract the ROI of the *original* face, then align the face
            # using facial landmarks
            (x, y, w, h) = rect_to_bb(rect)
            faceOrig = imutils.resize(image[y:y + h, x:x + w], width=256)
            # convert the landmark (x, y)-coordinates to a NumPy array
            shape = self.predictor(gray, rect)
            shape = shape_to_np(shape)
            # extract the left and right eye (x, y)-coordinates
            (lStart, lEnd) = self.FACIAL_LANDMARKS_IDXS["left_eye"]
            (rStart, rEnd) = self.FACIAL_LANDMARKS_IDXS["right_eye"]
            leftEyePts = shape[lStart:lEnd]
            rightEyePts = shape[rStart:rEnd]
            # compute
            leftEyeCenter = leftEyePts.mean(axis=0).astype("int")
            rightEyeCenter = rightEyePts.mean(axis=0).astype("int")
            # compute the angle between the eye centroids
            dY = rightEyeCenter[1] - leftEyeCenter[1]
            dX = rightEyeCenter[0] - leftEyeCenter[0]
            angle = np.degrees(np.arctan2(dY, dX)) - 180
            # compute the desired right eye x-coordinate based on the
            # desired x-coordinate of the left eye
            desiredRightEyeX = 1.0 - self.desiredLeftEye[0]
            # determine the scale of the new resulting image by taking
            # the ratio of the distance between eyes in the *current*
            # image to the ratio of distance between eyes in the
            # *desired* image
            dist = np.sqrt((dX ** 2) + (dY ** 2))
            desiredDist = (desiredRightEyeX - self.desiredLeftEye[0])
            desiredDist *= self.desiredFaceWidth
            scale = desiredDist / dist
            # compute center (x, y)-coordinates (i.e., the median point)
            # between the two eyes in the input image
            eyesCenter = ((leftEyeCenter[0] + rightEyeCenter[0]) // 2,
                          (leftEyeCenter[1] + rightEyeCenter[1]) // 2)
            # grab the rotation matrix for rotating and scaling the face
            M = cv2.getRotationMatrix2D(eyesCenter, angle, scale)
            # update the translation component of the matrix
            tX = self.desiredFaceWidth * 0.5
            tY = self.desiredFaceHeight * self.desiredLeftEye[1]
            M[0, 2] += (tX - eyesCenter[0])
            M[1, 2] += (tY - eyesCenter[1])
            # apply the affine transformation
            (w, h) = (self.desiredFaceWidth, self.desiredFaceHeight)
            output = cv2.warpAffine(image, M, (w, h),
                                    flags=cv2.INTER_CUBIC)
            return output
        except:
            pass
        return None


# In[ ]:


cam = cv2.VideoCapture(0)
#model = TheModelClass()
#model.load_state_dict(torch.load(PATH))
#model.eval()
FA = FaceAligner()
cv2.namedWindow("test")
while True:
    ret, frame = cam.read()
    image = FA.align(frame)
    cv2.imshow("test", image)
    if not ret:
        break
    k = cv2.waitKey(1)
    if k%256 == 27:
        # ESC pressed
        print("Escape hit, closing...")
        break
    elif k%256 == 32:
        # SPACE pressed
        img_name = "opencv_frame_{}.png".format(img_counter)
        cv2.imwrite(img_name, frame)
        print("{} written!".format(img_name))
        img_counter += 1
cam.release()
cv2.destroyAllWindows()


# Да нормально морды детектятся, чего вы начинаете?
# ![Мем2](images/wolf2.jpg)
