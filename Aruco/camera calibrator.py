# calibration.py
# Script to calibrate the camera using a chessboard pattern.
# Print a chessboard pattern (e.g., 9x6 squares) on paper, ensure it's flat.
# Run this script, show the chessboard to the webcam from various angles and distances.
# Press 's' to save a frame for calibration, collect 10-20 good frames.
# Press 'c' to calibrate once enough frames are collected.
# Press 'q' to quit.
# Saves camera_matrix and dist_coeffs to 'calibration.npz'.

############################################# < Required input variables > #############################################

CameraId = 0  # default cv2 camera index; override with --camera-id
PiCameraResolution = (1280, 800)  # size requested when --picamera is used

chessboard_size = (9, 6)  # Change if your chessboard is different (width, height) based on inner corners
square_size = 0.01985  # Size of each square in meters (adjust based on your print, e.g.,0.025 =  25mm)


import argparse
import time
import cv2
import numpy as np
from typing import List, Tuple
import numpy.typing as npt


class PiCameraCapture:
    """
    Minimal cv2.VideoCapture stand-in backed by picamera2.
    Camera matrix is invpixels, calibrate at the resolution intended for use.
    """

    def __init__(self, resolution=PiCameraResolution):
        from picamera2 import Picamera2

        self.picam2 = Picamera2()
        self.picam2.configure(self.picam2.create_preview_configuration(
            main={"format": "RGB888", "size": tuple(resolution)}))
        self.picam2.start()
        time.sleep(1)

    def isOpened(self):
        return True

    def read(self):
        return True, self.picam2.capture_array()

    def release(self):
        self.picam2.stop()
        self.picam2.close()


def open_camera(use_picamera: bool, camera_id: int):
    """Open the requested camera. Returns an object with read/isOpened/release.

    Uses cv2.VideoCapture by default to open videostream.

    Can optionally use the Raspberry PiCameras through picamera2.
    """
    if use_picamera:
        try:
            cap = PiCameraCapture()
        except ImportError as exc:
            raise SystemExit(
                f"--picamera needs picamera2, which is not importable here ({exc}).\n"
                f"On a non-Pi machine, drop --picamera to use cv2 camera {camera_id} instead."
            ) from None
        print(f"Using picamera2 at {PiCameraResolution[0]}x{PiCameraResolution[1]}")
        return cap

    print(f"Using cv2.VideoCapture({camera_id})")
    return cv2.VideoCapture(camera_id)


parser = argparse.ArgumentParser(
    description="Calibrate a camera from a chessboard pattern and save camera_matrix "
                "and dist_coeffs to calibration.npz.")
parser.add_argument("--camera-id", type=int, default=CameraId,
                    help=f"cv2 camera index to open (default: {CameraId})")
parser.add_argument("--picamera", action="store_true",
                    help="use the Raspberry Pi CSI camera via picamera2 instead of cv2. "
                         "Needed on a Pi: cv2 cannot reach the CSI camera through libcamera.")
args = parser.parse_args()


# Prepare object points (3D points in real world space)
objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)
objp *= square_size  # Scale by square size

# Arrays to store object points and image points from all images
objpoints: List[npt.NDArray[np.float32]] = []  # 3D points in real world space
imgpoints: List[npt.NDArray[np.float32]] = []  # 2D points in image plane

cap = open_camera(args.picamera, args.camera_id)

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