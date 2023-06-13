import cv2
import numpy as np

def Hough_Circles(img):
    # Process the image for circles using the Hough transform
    circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 2, img.shape[0] / 64, param1=24, param2=62,
                               minRadius=10, maxRadius=60)

    # Determine if any circles were found
    if circles is not None:
        if len(circles) > 1:
            print("False circles detected!")
        
        conf = 1.0 / len(circles)
        # convert the (x, y) coordinates and radius of the circles to integers
        circles = np.round(circles[0, :]).astype("int")

        # draw the circles
        x, y, r = circles[0]

        mask = np.zeros(img.shape)
        mask = cv2.circle(mask, (x, y), r, 255, -1)
        
    else:
        mask = np.zeros(img.shape)
        conf = 0.0

    return mask, conf

def Contours(img):
    contours = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]

    mask = np.zeros(img.shape)
    if len(contours) == 0:
        mask = np.zeros(img.shape)
        conf = 0.0
        return mask, conf
    max_contour = contours[0]
    max_area = cv2.contourArea(max_contour)

    for c in contours:
      area = cv2.contourArea(c)
      if area > max_area:
          max_area = area
          max_contour = c
    if max_area > 1800:
        # ellipse = cv2.fitEllipse(max_contour)
        # cv2.ellipse(mask, ellipse, (255, 0, 0), -1)
        cv2.drawContours(mask, [max_contour], -1, (255, 0, 0), -1)
        conf = 1.0
    else:
        mask = np.zeros(img.shape)
        conf = 0.0
    
    return mask, conf

def detect_pupil(img):
    # imgR = cv2.resize(img, (600, 600), interpolation=cv2.INTER_AREA)

    imgG = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thV, imgT = cv2.threshold(imgG, 30, 255, cv2.THRESH_BINARY)
    highTH = thV
    lowTH = thV / 2

    imgB = cv2.medianBlur(imgT, 7)

    # Find the binary image with edges from the thresholded image
    imgE = cv2.Canny(imgB, threshold1=lowTH, threshold2=highTH)

    mask, conf = Hough_Circles(imgE)
    mask, conf = Contours(imgE)
    

    return mask, conf
