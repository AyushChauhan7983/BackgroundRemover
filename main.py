import cv2
import cvzone
from cvzone.SelfiSegmentationModule import SelfiSegmentation
import os

cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)
cap.set(cv2.CAP_PROP_FPS, 30)
fpsReader = cvzone.FPS()
segmenter = SelfiSegmentation()

listImg = os.listdir("images")
imgList = []
for imgPath in listImg:
    img = cv2.imread(f'images/{imgPath}')
    imgList.append(img)

indexImg = 0

while True:
    success, img = cap.read()
    imgOut = segmenter.removeBG(img, imgList[indexImg], threshold=0.4)

    imgStacked = cvzone.stackImages([img, imgOut], 2, 1)
    fpsReader.update(imgStacked)
    cv2.imshow("Image", imgStacked)

    key = cv2.waitKey(1)
    if key == ord('a'):
        if indexImg > 0:
            indexImg -= 1
    elif key == ord('d'):
        if indexImg < len(imgList)-1:
            indexImg += 1
    elif key == ord('q'):
        break
