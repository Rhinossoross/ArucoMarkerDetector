"""Detect Aruco markers 2d relative rotations from any webcam 
   - requires OpenCV and Numpy
"""

#if running this program instead of importing, then these are the programs input variables:
#
cameraID:int = 0                            # id of camera
markerSize:float = 0.01985                  # size of the markers in meters (0.025 = 25mm)
expectedNumber:int = 4                      # expected number of markers to find
maxIterations:int=100                       # maximum number of frames it will attempt to find markers (only used if program not running in continuous mode)
continuous:bool=True                        # run the detection indefinitely
showVideo:bool=True                         # show the video frame
debugInfo:bool=True                         # output text
averagingItertions:int=15                   # the amount of frames it will try to average the angle over (values over 3 enable std based outlier removal)
calibrationFilePath:str = 'calibration.npz' # location of camera calibrator

# required imports
import cv2
import numpy as np
from os import system as clear_terminal  # Renamed for clarity
class _DetectedAruco: #the single underscore makes this a private class
    def __init__(self):
        self.corner = np.zeros((4, 2))
        self.id = None
        self.Update(self.corner)

    def Update(self, new_corners): 
        self.corner = new_corners
        self.Leftcenter = (self.corner[0] + self.corner[3]) / 2
        self.Rightcenter = (self.corner[1] + self.corner[2]) / 2
        self.CenterVector = self.Rightcenter - self.Leftcenter   # the vector maede from the center point of each virticle edge of the marker

def GetVectorAngle(vector1, vector2):
    """get the angle between two vectors in radians"""
    # The smallest angle between these two input vectors
    dot_product = np.dot(vector1, vector2)
    cross_product = vector1[0] * vector2[1] - vector1[1] * vector2[0]
    angle = np.arctan2(cross_product, dot_product)
    return angle

    #old method returning unsigned smallest angles
    dot_product = np.dot(vector1, vector2)
    norm_product = np.linalg.norm(vector1) * np.linalg.norm(vector2)
    if norm_product == 0:
        return 0
    angle = np.arccos(np.clip(dot_product / norm_product, -1.0, 1.0))
    return angle

class _CameraData:
    def __init__(self,cap,markerSize:float, detector: cv2.aruco.ArucoDetector,camera_matrix, dist_coeffs,object_points ):
        self.cap           = cap 
        self.markerSize    = markerSize
        self.detector      = detector
        self.camera_matrix = camera_matrix
        self.dist_coeffs   = dist_coeffs 
        self.object_points = object_points


def GetCameraData(
        cameraID:int = 0,
        markerSize:float = 0.02,
        calibrationFilePath:str = 'calibration.npz',
                ):
    """set up the camera for capture
    Parameters
    ----------
    CameraID : The numerical id of the camera
    markerSize : The size of the marker in meters (0.02 = 20mm)
    CalibraionPath : location of the calibration file.
    
    Returns
    -------
    success : true or false
    cameraData : object of all neccecary perminant data for aruco detection
    """
    # Create the ArUco dictionary and detector

    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
    detector_params = cv2.aruco.DetectorParameters()
    refine_params = cv2.aruco.RefineParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, detector_params, refine_params)

    # Open the webcam
    cap = cv2.VideoCapture(cameraID)
    if not cap.isOpened():
        if debugInfo: print("Error: Could not open video.")
        return False ,None

    # Get frame dimensions
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Load calibration if available, else use placeholder
    try:
        with np.load(calibrationFilePath) as data:
            camera_matrix = data['camera_matrix']
            dist_coeffs = data['dist_coeffs']
        if debugInfo: print("Loaded calibration from 'calibration.npz'")
    except FileNotFoundError:
        # Placeholder camera matrix and distortion coefficients
        focal_length = (frame_width + frame_height) / 2  # Rough estimate
        camera_matrix = np.array([[focal_length, 0, frame_width / 2],
                                [0, focal_length, frame_height / 2],
                                [0, 0, 1]], dtype=np.float32)
        dist_coeffs = np.zeros((4, 1), dtype=np.float32)
        if debugInfo: print("Using placeholder calibration (inaccurate); run calibration.py first for better results, or provide 'calibration.npz'.")

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
    return True, _CameraData(cap,markerSize,detector,camera_matrix,dist_coeffs,object_points)

def CleanupCamera(cameraData:_CameraData|None):
    """clean up camera when finished"""
    if cameraData != None:
        cameraData.cap.release()
    cv2.destroyAllWindows()

def GetMarkerAngle(
        CameraData:_CameraData|None,
        expectedNumber:int = 4,                  
        maxIterations:int=100,                   
        averagingItertions:int=15,               
        
        continuous:bool=False,                    
        showVideo:bool=False,                     
        debugInfo:bool=False,                     
        ) ->tuple[bool, list[float]]: 
    """
    Parameters
    ----------
    expectedNumber : expected number of markers to find
    maxIterations : maximum number of frames it is allwed to search for markers in (only used if program not running in continuous mode)
    averagingItertions : the amount of frames it will try to average the angle over (values over 3 enable std based outlier removal)
    calibrationFilePath : location of camera calibrator
    continuous : run the detection indefinitely
    showVideo : show the video frame
    debugInfo : print debug text

    Returns
    -------
    success : true/false dependent on if it picked up all expected markers (only used if program not running in continous mode)
    relativeAngles : list of the angles between markers, i.e x deg between 0&1, y deg between 1&2, etc.
    """

    if CameraData is None:
        return False, []
    if averagingItertions <1: 
        if debugInfo: print("Error: averagingIterations must be at least 1.\n set to 1")
        averagingItertions = 1

    

    ## mainloop set up
    angles = []
    outputAngles = []
    continueDetection = True
    markerFoundIterations = 0
    angelGroups = []

    ####### mainloop
    while continueDetection:
        maxIterations -= 1

        if maxIterations <= 0 and not continuous:
            continueDetection = False
            return False,[]
        ret, frame = CameraData.cap.read()
        if not ret:
            if debugInfo: print("Error: Could not read frame.")
            break
        cv2.flip(frame, 1)


        # Detect ArUco markers
        aruco_list = []
        corners, ids, rejected = CameraData.detector.detectMarkers(frame) # get detection from aruco library
        if ids is not None:
            for i in range(len(ids)):
                image_points = corners[i][0]  # 2D image points for this marker

                # Estimate pose using solvePnP
                success, rvec, tvec = cv2.solvePnP(CameraData.object_points, image_points, CameraData.camera_matrix, CameraData.dist_coeffs)
                if success:
                    detected_aruco = _DetectedAruco()
                    detected_aruco.id = ids[i][0]
                    detected_aruco.Update(image_points)
                    aruco_list.append(detected_aruco)

                    ### include 3D pose for this marker (translation and rotation vector) ###

                    #print(f"Marker ID {detected_aruco.id}:")
                    #print(f"  Translation (x, y, z): {tvec.flatten()}")
                    #print(f"  Rotation vector: {rvec.flatten()}")

                    # Optional: Draw axes on the frame for visualization
                    if showVideo: cv2.drawFrameAxes(frame, CameraData.camera_matrix, CameraData.dist_coeffs, rvec, tvec, CameraData.markerSize / 2)

            # Draw detected markers on the frame
            if showVideo: cv2.aruco.drawDetectedMarkers(frame, corners, ids)

        aruco_list.sort(key=lambda x: x.id if x.id is not None else 0)

        # compute angles between consecutive markers
        
        if len(aruco_list) > 1:
            if debugInfo: 
                try: clear_terminal('cls')  # Use 'clear' on Linux/macOS
                except: clear_terminal('clear')
            angles = []
            outputAngles = []
            for i in range(len(aruco_list) - 1):
                if aruco_list[i].id == aruco_list[i + 1].id - 1:
                    angle = GetVectorAngle(aruco_list[i].CenterVector, aruco_list[i + 1].CenterVector)
                    angles.append(f"{i}->{i+1}: {int(np.degrees(angle))} deg")
                    outputAngles.append(np.degrees(angle))
                    if debugInfo: print(f"Angle between marker {aruco_list[i].id} and marker {aruco_list[i + 1].id}: {int(np.degrees(angle))} degrees")
                else:
                    if debugInfo: print("missing marker",aruco_list[i].id + 1)
        if showVideo:
            for i in range(len(angles)):
                cv2.putText(frame, angles[i],
                            (10, 30 + i * 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        # Display the frame
            cv2.imshow('Frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        if len (angles) >= expectedNumber -1 and not continuous: ## if were not running continuously and we have found the expected number of markers
            if markerFoundIterations >= averagingItertions:                   # if we have got all the averaging iterations we need
                continueDetection = False              # stop detection    
                averagedAngles = []
                for i in range(expectedNumber - 1):
                    angle_list = [group[i] for group in angelGroups]
                    if len(angle_list) < 3:
                        # Too few for std-based rejection; use median as fallback
                        averagedAngles.append(np.median(angle_list))
                    else:
                        mean = np.mean(angle_list)
                        std = np.std(angle_list)
                        threshold = 1.3* std  # Tune this; lower for stricter rejection
                        filtered = [a for a in angle_list if abs(a - mean) <= threshold]
                        if not filtered:
                            filtered = angle_list  # Fallback if all rejected
                        averagedAngles.append(np.mean(filtered))
                if debugInfo: print("Averaged angles over",markerFoundIterations,"iterations:",averagedAngles)
                return True, averagedAngles
            markerFoundIterations +=1
            maxIterations +=1  # give another iteration to find more markers
            angelGroups.append(outputAngles)
        # Exit on 'q' key press    
    return False,[]
if __name__ == "__main__":
    success,data = GetCameraData(cameraID,markerSize,calibrationFilePath)
    if success:
        GetMarkerAngle(data, expectedNumber, maxIterations, averagingItertions, continuous, showVideo, debugInfo)
        CleanupCamera(data)