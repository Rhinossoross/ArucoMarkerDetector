# calibration.py
# Script to calibrate the camera using a chessboard pattern.
# Print a chessboard pattern (e.g., 9x6 squares) on paper, ensure it's flat.
# Run this script, show the chessboard to the webcam from various angles and distances.
# Press 's' to save a frame for calibration, collect 10-20 good frames.
# Press 'c' to calibrate once enough frames are collected.
# Press 'q' to quit.
# Saves camera_matrix and dist_coeffs to 'calibration.npz'.

############################################# < Required input variables > #############################################

CameraId = 0 

chessboard_size = (9, 6)  # Change if your chessboard is different (width, height) based on inner corners
square_size = 0.01985  # Size of each square in meters (adjust based on your print, e.g.,0.025 =  25mm)


import cv2
import numpy as np
from typing import List, Tuple
import numpy.typing as npt


# Prepare object points (3D points in real world space)
objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)
objp *= square_size  # Scale by square size

# Arrays to store object points and image points from all images
objpoints: List[npt.NDArray[np.float32]] = []  # 3D points in real world space
imgpoints: List[npt.NDArray[np.float32]] = []  # 2D points in image plane

cap = cv2.VideoCapture(CameraId)          


if not cap.isOpened():
    print("Error: Could not open video.")
    exit()
num_captured = 0
calibrated = False

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break
    cv2.flip(frame, 1)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    imageSize: Tuple[int, int] = gray.shape[::-1]
    
    ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)
    if ret:
        cv2.drawChessboardCorners(frame, chessboard_size, corners, ret)
        cv2.putText(frame, "Press 's' to save this frame for calibration", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.putText(frame, f"Captured: {num_captured}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    if num_captured >= 10 and not calibrated:
        cv2.putText(frame, "Press 'c' to calibrate", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imshow('Calibration', frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('s') and ret:
        objpoints.append(objp)
        imgpoints.append(corners.astype(np.float32))
        num_captured += 1
        print(f"Captured frame {num_captured}")
    elif key == ord('c') and num_captured >= 10:
        print("Calibrating...")
        
        ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints,imageSize, None, None) # type: ignore[call-arg]
        if ret:
            np.savez('calibration.npz', camera_matrix=camera_matrix, dist_coeffs=dist_coeffs)
            print("Calibration saved to 'calibration.npz'")
            calibrated = True
            break
        else:
            print("Calibration failed")

cap.release()
cv2.destroyAllWindows()