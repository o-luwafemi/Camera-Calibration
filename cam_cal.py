import numpy as np
import cv2 as cv
import glob

# chessboard inner corners
chessboard = (7, 7)

# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
 
# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((chessboard[0] * chessboard[1], 3), np.float32)
objp[:,:2] = np.mgrid[0:chessboard[0], 0:chessboard[1]].T.reshape(-1,2)
 
# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.
 
images = glob.glob('./data/*.jpg')

for i, fname in enumerate(images):
    
    img = cv.imread(fname)

    # Convert the image to grayscale
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
 
    # Find the chess board corners
    ret, corners = cv.findChessboardCorners(gray, chessboard, None)
 
    # If found, add object points, image points (after refining them)
    if ret == True:
        objpoints.append(objp)
 
        corners2 = cv.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
        imgpoints.append(corners2)
 
        # Draw and display the corners
        cv.drawChessboardCorners(img, (7,6), corners2, ret)
        cv.imshow('img', img)

        # Save images
        cv.imwrite(f'./output/calibrated/cal_img{i+1}.jpg', img)
        cv.waitKey(500)

    print(f'{i} of {len(images)}')

cv.destroyAllWindows()



# Calibrate the camera using the detected object points and image points

"""
ret: A flag indicating the success or failure of the camera calibration process. 
     It will be True if the calibration succeeds, and False otherwise.

mtx(camera matrix): A 3x3 matrix representing the camera intrinsic parameters. 
              These parameters include focal length, optical centers(principal point), and skew(camera distortion).
              
dist: A vector representing distortion coefficients. 
      These coefficients correct for radial and tangential lens distortion.
      
rvecs: A list of rotation vectors for each calibration image. 
       These vectors describe the rotation of the camera relative to the calibration pattern.
       
tvecs: A list of translation vectors for each calibration image. 
       These vectors describe the translation of the camera relative to the calibration pattern.

newCameraMatrix: A refined camera matrix optimized for undistortion. 
                 It adjusts the focal length and principal point to minimize distortion.
                 
roi: Region of interest (ROI) in the undistorted image. 
     It defines the coordinates of the rectangle containing the useful part of the undistorted image.

dst: The undistorted image obtained using the provided input image, 
     camera matrix, distortion coefficients, and refined camera matrix (newCameraMatrix).

"""

ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)


# Save intrinsic parameters
np.savez('output/parameters/intrinsic_params.npz', cameraMatrix=mtx, dist=dist)

# Estimate and save extrinsic parameters for each image
for i in range(len(images)):
    # Estimate extrinsic parameters for each image
    img = cv.imread(images[i])
    _, rvec, tvec = cv.solvePnP(objpoints[i], imgpoints[i], mtx, dist)
   

    # Save extrinsic parameters
    np.savez(f'./output/parameters/extrinsic_params_{i+1}.npz', rvec=rvec, tvec=tvec)


# Undistort the input image using the camera calibration parameters
img = cv.imread('./data/img1.jpg')
h,  w = img.shape[:2]
newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))


# undistort
dst = cv.undistort(img, mtx, dist, None, newcameramtx)
 
# crop the image
x, y, w, h = roi
dst = dst[y:y+h, x:x+w]
cv.imwrite('./output/undistorted/calibresult1.jpg', dst)



# undistort the input image using remapping
mapx, mapy = cv.initUndistortRectifyMap(mtx, dist, None, newcameramtx, (w,h), 5)
dst = cv.remap(img, mapx, mapy, cv.INTER_LINEAR)
 
# crop the image
x, y, w, h = roi
dst = dst[y:y+h, x:x+w]
cv.imwrite('./output/undistorted/calibresult2.jpg', dst)


#  Re-projection Error
mean_error = 0
for i in range(len(objpoints)):
    imgpoints2, _ = cv.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
    error = cv.norm(imgpoints[i], imgpoints2, cv.NORM_L2)/len(imgpoints2)
    mean_error += error
 
print( "total error: {}".format(mean_error/len(objpoints)) )