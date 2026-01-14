from ArucoDetector import GetMarkerAngle, GetCameraData, CleanupCamera

GotData,CameraData = success,data = GetCameraData(cameraID =0)

if GotData:
    for x in range (10):
        GotAngles,angles = GetMarkerAngle(CameraData, maxIterations=10, averagingItertions=5)
        if GotAngles:
            for i in range(len(angles)):
                print(f"Angle between marker {i} & {i+1}: {angles[i]} degrees")
        else:
            print("No markers detected.")
        print("#################################")
CleanupCamera(CameraData)