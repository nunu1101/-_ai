# STEP 1 : import modules
import argparse
import cv2
import sys
import numpy as np
import insightface
from insightface.app import FaceAnalysis
from insightface.data import get_image as ins_get_image

# STEP 2 : create interface instance
app = FaceAnalysis()
app.prepare(ctx_id=0, det_size=(640,640))

# STEP 3 : load input image
img1 = cv2.imread("riize.jpg")
img2 = cv2.imread("riize2.jpg")

# img = ins_get_image('t1')

# STEP 4 : inference
faces1 = app.get(img1)
faces2 = app.get(img2)
print(len(faces1))
print(len(faces2))

# STEP 5 : draw detection result
# then print all-to-all face similarity

feat1 = np.array(faces1[1].normed_embedding, dtype=np.float32)
feat2 = np.array(faces2[0].normed_embedding, dtype=np.float32)
sims = np.dot(feat1, feat2.T)
print(sims)

# assert len(faces1)==6
# assert len(faces2)==1

# print(faces)

# STEP 5 : draw detection result
# rimg = app.draw_on(img, faces)
# # cv2.imshow("test", rimg)
# # cv2.waitKey(0)
# cv2.imwrite("./t1_output.jpg", rimg)

# # then print all-to-all face similarity
# feats = []
# for face in faces:
#     feats.append(face.normed_embedding)
# feats = np.array(feats, dtype=np.float32)
# sims = np.dot(feats, feats.T)
# print(sims)

