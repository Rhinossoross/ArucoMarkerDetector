import cv2
from cv2 import aruco

# save markers as images

arucoDict = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
dictionarySize = 6 # 6x6
amountOfmarkers = 5 # number of markers to generate
markerSize = dictionarySize+2  # size of the marker image in pixels 
for i in range(amountOfmarkers):
    markerImage = aruco.generateImageMarker(arucoDict, i, markerSize)
    cv2.imwrite("marker maker/marker_{}.png".format(i), markerImage)

print("when loading these markers to a finger, go from the base to the finger tip in order of marker 0 to marker 4")
