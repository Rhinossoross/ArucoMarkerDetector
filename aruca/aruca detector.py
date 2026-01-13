# Updated main.py
# Changes:
# - Replaced cv2.aruco.estimatePoseSingleMarkers with manual cv2.solvePnP for compatibility.
# - Load camera_matrix and dist_coeffs from 'calibration.npz' if exists, else use placeholder.
# - Added object points definition for ArUco markers (assuming square markers, top-left clockwise).

import time
import cv2
import numpy as np
from os import system as clear_terminal  # Renamed for clarity

class DetectedAruco:
    def __init__(self):
        self.corner = np.zeros((4, 2))
        self.id = None
        self.Update(self.corner)
        self.ColisionPoints = []  # Assuming this is intentional; not used in provided code

    def Update(self, new_corners): 
        self.corner = new_corners
        self.Leftcenter = (self.corner[0] + self.corner[3]) / 2
        self.Rightcenter = (self.corner[1] + self.corner[2]) / 2
        self.CenterVector = self.Rightcenter - self.Leftcenter

def GetVectorAngle(vector1, vector2):
    # The smallest angle between these two input vectors
    dot_product = np.dot(vector1, vector2)
    norm_product = np.linalg.norm(vector1) * np.linalg.norm(vector2)
    if norm_product == 0:
        return 0
    angle = np.arccos(np.clip(dot_product / norm_product, -1.0, 1.0))
    return angle

# Create the ArUco dictionary and detector
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
detector_params = cv2.aruco.DetectorParameters()
refine_params = cv2.aruco.RefineParameters()
detector = cv2.aruco.ArucoDetector(aruco_dict, detector_params, refine_params)

# Open the webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Get frame dimensions
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Load calibration if available, else use placeholder
try:
    with np.load('calibration.npz') as data:
        camera_matrix = data['camera_matrix']
        dist_coeffs = data['dist_coeffs']
    print("Loaded calibration from 'calibration.npz'")
except FileNotFoundError:
    # Placeholder camera matrix and distortion coefficients
    focal_length = (frame_width + frame_height) / 2  # Rough estimate
    camera_matrix = np.array([[focal_length, 0, frame_width / 2],
                              [0, focal_length, frame_height / 2],
                              [0, 0, 1]], dtype=np.float32)
    dist_coeffs = np.zeros((4, 1), dtype=np.float32)
    print("Using placeholder calibration (inaccurate); run calibration.py first for better results")

# Marker size in meters (1cm = 0.01m)
marker_size = 0.01

# Define 3D object points for a single ArUco marker (corners in marker coordinate system)
half_size = marker_size / 2
object_points = np.array([
    [-half_size, half_size, 0],   # Top-left
    [half_size, half_size, 0],    # Top-right
    [half_size, -half_size, 0],   # Bottom-right
    [-half_size, -half_size, 0]   # Bottom-left
], dtype=np.float32)
angles = []
while True:
    time.sleep(0.5)
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break
    # Detect ArUco markers
    aruco_list = []
    corners, ids, rejected = detector.detectMarkers(frame)
    if ids is not None:
        for i in range(len(ids)):
            image_points = corners[i][0]  # 2D image points for this marker

            # Estimate pose using solvePnP
            success, rvec, tvec = cv2.solvePnP(object_points, image_points, camera_matrix, dist_coeffs)
            if success:
                detected_aruco = DetectedAruco()
                detected_aruco.id = ids[i][0]
                detected_aruco.Update(image_points)
                aruco_list.append(detected_aruco)

                #rint 3D pose for this marker (translation and rotation vector)
                #print(f"Marker ID {detected_aruco.id}:")
                #print(f"  Translation (x, y, z): {tvec.flatten()}")
                #print(f"  Rotation vector: {rvec.flatten()}")

                # Optional: Draw axes on the frame for visualization
                cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, rvec, tvec, marker_size / 2)

        # Draw detected markers on the frame
        cv2.aruco.drawDetectedMarkers(frame, corners, ids)

    # Sort the list by marker ID for consistent angle calculation (assuming IDs are sequential for "adjacent")
    # Alternatively, sort by x-position: aruco_list.sort(key=lambda x: np.mean(x.corner[:, 0]))
    aruco_list.sort(key=lambda x: x.id if x.id is not None else 0)

    # compute angles between consecutive markers
    
    if len(aruco_list) > 1:
        clear_terminal('cls')  # Use 'clear' on Linux/macOS
        angles = []
        for i in range(len(aruco_list) - 1):
            if aruco_list[i].id == aruco_list[i + 1].id - 1:
                angle = GetVectorAngle(aruco_list[i].CenterVector, aruco_list[i + 1].CenterVector)
                angles.append(f"{i}->{i+1}: {int(np.degrees(angle))} deg")
                print(f"Angle between marker {aruco_list[i].id} and marker {aruco_list[i + 1].id}: {int(np.degrees(angle))} degrees")
            else:
                print("missing marker",aruco_list[i].id + 1)
    for i in range(len(angles)):
        cv2.putText(frame, angles[i],
                    (10, 30 + i * 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    # Display the frame
    cv2.imshow('Frame', frame)

    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()