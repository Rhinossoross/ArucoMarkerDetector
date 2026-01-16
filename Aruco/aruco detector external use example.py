from ArucoDetector import GetMarkerAngle, GetCameraData, CleanupCamera
# set up camera
GotData,CameraData = GetCameraData(cameraID =0)
if GotData:
    for x in range (1):
        # run program in headless mode looking for 4 markers trying to get 15 values for each angle to average over 
        GotAngles,angles = GetMarkerAngle(CameraData,5, maxIterations=100, averagingItertions=15) 
        if GotAngles:
            #process angles
           for i in range(len(angles)):
               print(f"Angle between marker {i} & {i+1}: {angles[i]} degrees")
        else:
            print("No markers detected.")
        print("#################################")
CleanupCamera(CameraData)