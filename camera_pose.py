import numpy as np
import cv2 as cv
import glob


# chessboard inner corners
chessboard = (7, 7)


# Load previously saved data
with np.load('output/parameters/intrinsic_params.npz') as X:
    # mtx, dist, _, _ = [X[i] for i in ('mtx','dist','rvecs','tvecs')]
    mtx, dist = [X[i] for i in ('mtx', 'dist')]

# The draw function takes the corners in the chessboard
# (obtained using cv.findChessboardCorners()) and axis points to draw a 3D axis. 
def draw(img, corners, imgpts):
    imgpts = np.int32(imgpts).reshape(-1,2)

    corner = tuple(corners[0].ravel())

    # Draw 3D Axis
    # img = cv.line(img, corner, tuple(imgpts[0].ravel()), (255,0,0), 5)
    # img = cv.line(img, corner, tuple(imgpts[1].ravel()), (0,255,0), 5)
    # img = cv.line(img, corner, tuple(imgpts[2].ravel()), (0,0,255), 5)



    # Draw cube
    # draw ground floor in green
    img = cv.drawContours(img, [imgpts[:4]],-1,(0,255,0),-3)
 
    # draw pillars in blue color
    for i,j in zip(range(4),range(4,8)):
        img = cv.line(img, tuple(imgpts[i]), tuple(imgpts[j]),(255),3)
 
    # draw top layer in red color
    img = cv.drawContours(img, [imgpts[4:]],-1,(0,0,255),3)

    return img

# Termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
objp = np.zeros((chessboard[0] * chessboard[1], 3), np.float32)
objp[:,:2] = np.mgrid[0:chessboard[0], 0:chessboard[1]].T.reshape(-1,2)
 
# axis = np.float32([[3,0,0], [0,3,0], [0,0,-3]]).reshape(-1,3)
axis = np.float32([[0,0,0], [0,3,0], [3,3,0], [3,0,0],
                   [0,0,-3],[0,3,-3],[3,3,-3],[3,0,-3] ])




for i, fname in enumerate(glob.glob('./data/*.jpg')):
    img = cv.imread(fname)
    gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    ret, corners = cv.findChessboardCorners(gray, chessboard, None)
 
    if ret == True:
        corners2 = cv.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
 
        # Find the rotation and translation vectors.
        ret,rvecs, tvecs = cv.solvePnP(objp, corners2, mtx, dist)
 
        # project 3D points to image plane
        imgpts, jac = cv.projectPoints(axis, rvecs, tvecs, mtx, dist)
 
        img = draw(img, corners2, imgpts)
        cv.imshow('img',img)
        cv.imwrite(f'./output/pose/pos_img{i+1}.jpg', img)
        cv.waitKey(1000)
 
cv.destroyAllWindows()

