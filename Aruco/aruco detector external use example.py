from ArucoDetector import GetMarkerAngle, GetCameraData, CleanupCamera
# set up camera
GotData,CameraData = GetCameraData(cameraID =0)
if GotData:
    for x in range (10):
        GotAngles,angles = GetMarkerAngle(CameraData, maxIterations=10, averagingItertions=5)
        if GotAngles:
            #process angles
            for i in range(len(angles)):
                print(f"Angle between marker {i} & {i+1}: {angles[i]} degrees")
        else:
            print("No markers detected.")
        print("#################################")
CleanupCamera(CameraData)