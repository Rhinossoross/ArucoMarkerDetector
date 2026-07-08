"""Detect Aruco markers 2d relative rotations from any webcam 
   - requires OpenCV and Numpy
"""

#if running this program instead of importing, then these are the programs input variables:
#
cameraID:int = 0                            # id of camera
markerSize:float = 0.01985                  # size of the markers in meters (0.025 = 25mm)
expectedNumber:int = 4                      # expected number of markers to find
maxIterations:int=100                       # maximum number of frames it will attempt to find markers (only used if program not running in continuous mode)
averagingItertions:int=15                   # the amount of frames it will try to average the angle over (values over 3 enable std based outlier removal)
OutlierRejectionThreshold:float = 0.8       # how strictly to reject outliers when > 3 averageing frames have been found (lower is harsher)
ParralellToCamera = False                   # is marker parralel to camera (non paralell use 3d rotation matries which take more processing)
continuous:bool=True                        # run the detection indefinitely
showVideo:bool=True                         # show the video frame
debugInfo:bool=False                        # output text
calibrationFilePath:str = 'calibration.npz' # location of camera calibrator

# required imports
import cv2
import numpy as np
class _DetectedAruco: #the single underscore makes this a private class
    def __init__(self):
        self.corner = np.zeros((4, 2))
        self.id = None
        self.Update(self.corner)
        self.rotM = None
        self.rotV = None

    def Update(self, new_corners): 
        self.corner = new_corners
        self.Leftcenter = (self.corner[0] + self.corner[3]) / 2
        self.Rightcenter = (self.corner[1] + self.corner[2]) / 2
        self.CenterVector = self.Rightcenter - self.Leftcenter   # the vector maede from the center point of each virticle edge of the marker

def GetVectorAngle(vector1, vector2):
    """get the smallest angle between two vectors in radians"""
    dot_product = np.dot(vector1, vector2)
    cross_product = vector1[0] * vector2[1] - vector1[1] * vector2[0]
    angle = np.arctan2(cross_product, dot_product)
    return angle

def GetRotMatAngles(RotationMtrix1,RotationMatrix2):
    """get the relative rotations between two 3x3 rotation matrix"""
    RelativeAngleMatrix = np.dot(RotationMtrix1.T,RotationMatrix2)
    RzMat = np.ones((3,3))
    cv2.RQDecomp3x3(RelativeAngleMatrix, Qz = RzMat)
    anglez = np.atan2(RzMat[1][0],RzMat[0][0])
    return anglez

class _CameraData:
    def __init__(self,cap,markerSize:float, detector: cv2.aruco.ArucoDetector,camera_matrix, dist_coeffs,object_points ):
        self.cap           = cap 
        self.markerSize    = markerSize
        self.detector      = detector
        self.camera_matrix = camera_matrix
        self.dist_coeffs   = dist_coeffs 
        self.object_points = object_points


def GetCameraData(
        cap:cv2.VideoCapture = None,
        markerSize:float = 0.01,
        calibrationFilePath:str = 'calibration.npz',
                ):
    """set up the camera for capture
    Parameters
    ----------
    CameraID : The numerical id of the camera
    markerSize : The size of the marker in meters (0.01 = 10mm)
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
        OutlierRejectionThreshold:float = 1.2,               
        ParralellToCamera = False,
        continuous:bool=False,                    
        showVideo:bool=False,                     
        debugInfo:bool=False,                     
        ) ->tuple[bool, list[float]]: 
    """
    Parameters
    ----------
    expectedNumber : expected number of markers to find
    maxIterations : maximum number of frames it is allwed to search for markers in (only used if program not running in continuous mode)
    averagingItertions : the amount of frames it will try to average the angle over (values over 3 enable MAD based outlier removal)
    calibrationFilePath : location of camera calibrator
    ParralellToCamera : is marker parralel to camera (non paralell use 3d rotation matrixes which take more processing)
    OutlierRejectionThreshold : how strictly to reject outliers when > 3 averageing frames have been found (lower is harsher)
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
    np.set_printoptions(legacy='1.25')
    

    ## mainloop set up
    markerFoundIterations = 0
    angelGroups = []
    ####### mainloop
    if debugInfo: print("\n\n ######################## Detection started ################### \n")
    while 1:
        if not continuous:
            maxIterations -= 1

        ret, frame = CameraData.cap.read()
        if not ret:
            if debugInfo: print("Error: Could not read frame.")
            break
        if showVideo: cv2.flip(frame, 1)
        if debugInfo: 
            if not continuous: print(f"\n ###### frame {maxIterations} ########### \n")
            else: print(f"###### next frame ##########")


        # Detect ArUco markers
        angles = []
        outputAngles = []
        aruco_list = []
        corners, ids, rejected = CameraData.detector.detectMarkers(frame) # get detection from aruco library
        if ids is not None:
            for i in range(len(ids)):
                image_points = corners[i][0]  # 2D image points for this marker

                # Estimate pose using solvePnP
                success, rvec, tvec = cv2.solvePnP(CameraData.object_points, image_points, CameraData.camera_matrix, CameraData.dist_coeffs, flags=cv2.SOLVEPNP_IPPE_SQUARE)
                if success:
                    detected_aruco = _DetectedAruco()
                    detected_aruco.id = ids[i][0]
                    detected_aruco.Update(image_points)
                    aruco_list.append(detected_aruco)

                    ### include 3D pose for this marker (translation and rotation vector) ###
                    detected_aruco.rotV = rvec.flatten()
                    rotM = cv2.Rodrigues(src = rvec.flatten())[0]
                    detected_aruco.rotM = rotM

                    # Optional: Draw axes on the frame for visualization
                    if showVideo: cv2.drawFrameAxes(frame, CameraData.camera_matrix, CameraData.dist_coeffs, rvec, tvec, CameraData.markerSize / 2)

            # Draw detected markers on the frame
            if showVideo: cv2.aruco.drawDetectedMarkers(frame, corners, ids)

        aruco_list.sort(key=lambda x: x.id if x.id is not None else 0)

        # compute angles between consecutive markers

        if len(aruco_list) > 1:
            for i in range(len(aruco_list) - 1):
                if aruco_list[i].id == aruco_list[i + 1].id - 1:
                    vecAngle = GetVectorAngle(aruco_list[i].CenterVector, aruco_list[i + 1].CenterVector)
                    rotMatAngle = GetRotMatAngles(aruco_list[i].rotM, aruco_list[i+1].rotM)
                    if ParralellToCamera: 
                        angles.append(f"{i}->{i+1}: {int(np.degrees(vecAngle))} deg")
                        outputAngles.append(np.degrees(vecAngle))    # vecAngle is more acurate when parralel to camera
                        if debugInfo: print(f"Angle between marker {aruco_list[i].id} and marker {aruco_list[i + 1].id} \n from rotMat: {int(np.degrees(vecAngle))} degrees \n from averaged vector Angle:{int(np.degrees(vecAngle))}")
                    else: 
                        angles.append(f"{i}->{i+1}: {int(np.degrees(rotMatAngle))} deg")
                        outputAngles.append(np.degrees(rotMatAngle)) # rotMatAngle is more acurate when non parralel to camera
                        if debugInfo: print(f"Angle between marker {aruco_list[i].id} and marker {aruco_list[i + 1].id} \n from rotMat: {int(np.degrees(rotMatAngle))} degrees \n from averaged vector Angle:{int(np.degrees(vecAngle))}")
                else:
                    if debugInfo: print("missing marker",aruco_list[i].id + 1)

        ## showing video
        if showVideo:
            for i in range(len(angles)):
                cv2.putText(frame, angles[i],
                            (10, 30 + i * 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.imshow('Frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        ## outlier rejection and averaging
        if not continuous and (len (angles) >= expectedNumber -1 or maxIterations == 1) : ## if we're not running continuously and we have found the expected number of markers or run out of iterations
            if markerFoundIterations >= averagingItertions or maxIterations == 1:                   # if we have got all the averaging iterations we need
                averagedAngles = []
                if debugInfo: print("\n\n ######################## Detection finished ################### \n  ############# beginning averaging: ")
                for i in range(expectedNumber - 1):
                    angle_list = [group[i] for group in angelGroups]
                    if len(angle_list) <= 0: 
                        if debugInfo: print("no markers found")
                        return False, []
                    elif len(angle_list) < 3:
                        # Too few for std-based rejection; use median as fallback
                        averagedAngles.append(np.median(angle_list))
                        if debugInfo: print(f"{len(angle_list)} frames found for marker {i}->{i+1} calculating median: \n angle list:{angle_list} \n avg angle:{np.median(angle_list)}")
                    else:
                        median = np.median(angle_list)
                        MedianAbsoluteDeviations = np.median(np.abs(np.array(angle_list) - median))
                        threshold = OutlierRejectionThreshold * MedianAbsoluteDeviations  # Tune this; lower for stricter rejection
                        filtered = [a for a in angle_list if abs(a - median) <= threshold]
                        if not filtered:
                            filtered = angle_list  # Fallback if all rejected
                        averagedAngles.append(np.mean(filtered))
                        if debugInfo: print(f"{len(angle_list)} frames found for marker {i}->{i+1} performing std based rejecton \n angle list:{angle_list } \nmedian:{median}\nMedian Absolute Deviations:{MedianAbsoluteDeviations}\n exclusion threshold:{threshold}\n filtered list:{filtered} \n mean angle:{np.mean(filtered)}")
                if debugInfo: print("Averaged angles over",markerFoundIterations,"iterations:",averagedAngles)
                return True, averagedAngles
            
            # if we dont have everything we need
            markerFoundIterations +=1
            maxIterations +=1  # give another iteration to find more markers
            angelGroups.append(outputAngles)
        # Exit on 'q' key press    
    return False,[]
if __name__ == "__main__":
    success,data = GetCameraData(cameraID,markerSize,calibrationFilePath)
    if success:
        GetMarkerAngle(data, expectedNumber, maxIterations, averagingItertions,OutlierRejectionThreshold,ParralellToCamera, continuous, showVideo, debugInfo)
        CleanupCamera(data)