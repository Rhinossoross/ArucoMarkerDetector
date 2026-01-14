from ArucoDetector import GetMarkerAngle

success, angles = GetMarkerAngle(cameraID = 0,showVideo=True, averagingItertions=10, continuous=False)
if success:
    for i in range(len(angles)):
        print(f"Angle between marker {i} & {i+1}: {angles[i]} degrees")
else:
    print("No markers detected.")