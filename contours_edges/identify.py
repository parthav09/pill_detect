import cv2
import time
from argparse import ArgumentParser
import imutils

# Using command line arguments for importing mages
z=0
start = time.time() # starting the timer to time the process

ap = ArgumentParser()
ap.add_argument("--pills", "-i", help="Path to the image", required=True)
args = vars(ap.parse_args())

pills = cv2.imread(args["pills"]) # reading the image
gray = cv2.cvtColor(pills, cv2.COLOR_BGR2GRAY) # convert the image to a grayscale image
# blurred = cv2.GaussianBlur(gray, (9,9), 0) # blurring

# using thresholding to convert to binary
(T, threshinv) = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

# using sobel for gradient detection
xaxis = cv2.Sobel(threshinv, ddepth=cv2.CV_64F, dx=1, dy=0)  # identifying horizontal changes
yaxis = cv2.Sobel(threshinv, ddepth=cv2.CV_64F, dx=0, dy=1)  # identifying vertical changes

# calculating the absolute value of gradient images
xaxis = cv2.convertScaleAbs(xaxis)
yaxis = cv2.convertScaleAbs(yaxis)

combinedpills = cv2.addWeighted(xaxis, 0.5, yaxis, 0.5, 0)# Gx + Gy

cv2.imshow('Combined: ', combinedpills)  # combined images
cv2.imshow("Threshed: ", threshinv)  # thresholded images

# finding the contours
contours = cv2.findContours(combinedpills.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contours = imutils.grab_contours(contours)
print(len(contours))

# removing unnesecary contours
for (i, c) in enumerate(contours):
    area = cv2.contourArea(c)
    if area > 1000:
        z += 1
        print("Found {} contours".format(z))
        (x, y, w, h) = cv2.boundingRect(c)
        (centerx, centery) = (x + (w // 2), y + (h // 2))
        (b, g, r) = pills[centery, centerx]
        print(b,g,r)
        if (r > 200 and b <50 and g>200):
            cv2.drawContours(pills, c, -1, (0, 0, 255), 2)
        else:
            cv2.drawContours(pills, c, -1, (0, 225, 0), 2)

cv2.imshow("Clone: ", pills)
end = time.time()
time_lapsed = end - start
totaltime = time_lapsed % 60
print(totaltime)
cv2.waitKey(100000)
# python3 identify.py --pills White_1.bmp