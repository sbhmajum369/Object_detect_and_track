
import numpy as np
import cv2


# cap = cv2.VideoCapture(0)
img1=cv2.imread('./Pictures/object.jpg',0)
outimg=np.zeros_like(img1)

orb=cv2.ORB_create()

# while True:
kp1, des1 = orb.detectAndCompute(img1,None)
frame=cv2.imread('./Pictures/whole.jpg')# _,frame = cap.read()
img2 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

kp2, des2 = orb.detectAndCompute(img2,None)
## create BFMatcher object
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

## Match descriptors.
matches = bf.match(des1,des2)

## Sort them in the order of their distance.
matches = sorted(matches, key = lambda x:x.distance)
L=len(matches)

if L>30:
	x=0.3
else:
	x=0.5

n=int(x*L)	## Number of top matches to consider

mat=matches[:n]
good=[]
for m in mat:
	good.append(m)

src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
matchesMask = mask.ravel().tolist()
h,w = img1.shape
pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
dst = cv2.perspectiveTransform(pts,M)

img2 = cv2.polylines(img2,[np.int32(dst)],True,4,4, cv2.LINE_AA)
draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                   singlePointColor = None,
                   matchesMask = matchesMask, # draw only inliers
                   flags = 2)

## To draw the matched keypoints between the images
# img4 = cv2.drawMatches(img1,kp1,img2,kp2,good,None,**draw_params)

cv2.namedWindow('Matched Image', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Matched Image', 720, 480)
cv2.imshow('Matched Image',img2)
# k = cv2.waitKey(30) & 0xff
# if k == 27:
# 	break

cv2.waitKey()
cv2.destroyAllWindows()
# cap.release()
