from ArucoDetector import GetMarkerAngle, GetCameraData, CleanupCamera
import cv2
# set up camera
GotData,CameraData = GetCameraData(cap =cv2.VideoCapture(0))


if GotData:
    for x in range (1):
        # run program in headless mode looking for 4 markers trying to get 15 values for each angle to average over 
        GotAngles,angles = GetMarkerAngle(CameraData,4, maxIterations=100, averagingItertions=15) 
        if GotAngles:
            #process angles in some way
           for i in range(len(angles)):
               print(f"Angle between marker {i} & {i+1}: {angles[i]} degrees")
        else:
            print("No markers detected.")
        print("#################################")
        
#cleanup camera
CleanupCamera(CameraData)