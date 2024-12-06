"""
    Area Detect Functions Author(s): Lisette Ruano
    Probe Detection Author(s): Lisette Ruano
    coordsDiff Author(s): Lisette Ruano
        
    Commenting/Code Structure was implemented by Lisset Rico.
        
    Collaborator(s): Argonne National Laboratory (Nazar Delegan, Clayton DeVault), Break Through Tech (Kyle Cheek)
    Date Created: 06/26/2024
"""

import time
import cv2
import requests
import json
import os
import random
import numpy as np


"""
    
    areaDetectNonColor : Detects the percentage of the unetched area of a given membrane
    
    Args:
        img_path: string
    Returns:
        whole_number_percentage: integer
    Raises:
        None.
    
"""

def areaDetectNonColor(img_path:str):
    # Read in image location
    image = cv2.imread(img_path)

    # Converts image to gray scale and blurs it
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.medianBlur(gray, 5)
    
    # Sharpens the blurred image
    sharpen_kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    sharpen = cv2.filter2D(blur, -1, sharpen_kernel)
    
    # Setting color threshold and cleaning up noise in the picture
    thresh = cv2.threshold(blur, 148, 255, cv2.THRESH_BINARY_INV)[1]
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    close = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
    black_threshold = 50

    # Counting black pixels and total pixels
    black_pixels = np.count_nonzero(close < black_threshold)
    total_pixels = close.size

    # Calculates percentage of black pixels then shows altered pictures
    percentage_black = (black_pixels / total_pixels) * 100
    whole_number_percentage = int(percentage_black)

    return whole_number_percentage

"""
    
    areaDetectColorBinary : Detects the percentage of the unetched area of a given colored membrane
    
    Args:
        img_path: string
    Returns:
        whole_number_percentage: integer
    Raises:
        None.
    
"""
def areaDetectColorBinary(img_path:str):

    # read in image location
    image = cv2.imread(img_path)

    # converts image to gray scale and blurs it
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.medianBlur(gray, 5)
    
    # Sharpens the blurred image
    sharpen_kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    sharpen = cv2.filter2D(blur, -1, sharpen_kernel)
    

    ret3,otsu = cv2.threshold(sharpen,35,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    close = cv2.morphologyEx(otsu, cv2.MORPH_CLOSE, kernel, iterations=2)
    black_threshold = 50

    
    # Counting black pixels and total pixels
    black_pixels = np.count_nonzero(close < black_threshold)
    
    total_pixels = otsu.size

    # Calculates percentage of black pixels then shows altered pictures
    percentage_black = (black_pixels / total_pixels) * 100
    whole_number_percentage = int(percentage_black)
    cv2.imshow('close', close) # this shoes black and white pixels
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return whole_number_percentage


"""

    probe_detection : Detects the probes

    Args:
        img_path: string
    Returns:
        detected: boolean -> true if a square is found, false otherwise
        rightProbe: array -> probe coordinates
        leftProbe: array -> probe coordinates
    Raises:
        None
            
"""
def probe_detection(img_path):
    # initialize variables, read image
    detected = False
    image = cv2.imread(img_path)
    alpha = 2.5 
    beta = 30
    
    img = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    
    #Converts picture into grayscale and blurs it
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
    blur = cv2.GaussianBlur(gray,(5,5),0) 

    #Apply otsu threshold
    ret3,otsu = cv2.threshold(blur,35,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    image_binary = cv2.bitwise_not(otsu)

    (contours,_) = cv2.findContours(image_binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    count = 0
    
    for cnt in contours:
        approx = cv2.approxPolyDP(cnt, 0.1*cv2.arcLength(cnt, True), True)
        area = cv2.contourArea(cnt) #Calculates area of objects to disregard stray small shapes it finds
        
        if len(approx) == 3 and area > 1000:
            
            # Extract vertice of the triangle
            point1 = tuple(approx[0][0])
            point2 = tuple(approx[1][0])
            point3 = tuple(approx[2][0])

            verticesList = [point1,point2,point3]

            tipofProbe = None

            #need to find lowest x for right probe
            if count == 0: 
                # lowest_x_value = float('inf') 
                # for vertex in verticesList:
                #     x_value = vertex[0]  # Get x coordinate of the vertex
    
                #     # Compare x_value with highest_x_value found so far
                #     if x_value < lowest_x_value:
                #         lowest_x_value = x_value
                #         tipofProbe = vertex
                # rightProbe = tipofProbe
                lowest_y_value = float('inf')
                for vertex in verticesList:
                    y_value = vertex[1]
                    if y_value < lowest_y_value:
                        lowest_y_value = y_value
                        tipofProbe = vertex
                rightProbe = tipofProbe
            # need to find highest x for left probe
            else:
                 highest_x_value = -float('inf') 
                 for vertex in verticesList:
                    x_value = vertex[0]  # Get x coordinate of the vertex
    
                    # Compare x_value with highest_x_value found so far
                    if x_value > highest_x_value:
                        highest_x_value = x_value
                        tipofProbe = vertex
                 leftProbe = tipofProbe
            
            img = cv2.drawContours(image, [cnt], -1, (0,255,255), 2)

            cv2.circle(img, tipofProbe, 5, (0, 255, 0), -1)  # Green dot at point
            
            M = cv2.moments(cnt)
            count += 1

    cv2.imshow('final',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    if count == 2:
        detected = True

    return detected,rightProbe,leftProbe


"""

    coordsDiff : Obtains coordinates of probes and membranes in order to see how close or far they are
    by subtracting them from each other

    Args:
        img_path: string
    Returns:
        leftX: integer -> coordinate
        leftY: integer -> coordinate
        rightX: integer -> coordinate
        rightY: integer -> coordinate
    Raises:
        None
            
"""


def coordsDiff(img_path):
    detected,rightProbe,leftProbe = probe_detection(img_path)
    square,bR,uL = square_detect(img_path)

    print("Upper Left Corner: ", uL)
    print("Bottom Right Corner: ",bR,"\n")

    print("Right Probe:",rightProbe)
    print("Left Probe:", leftProbe, "\n")
    
    leftX = abs(leftProbe[0]-uL[0])
    leftY = abs(leftProbe[1]-uL[1])
    rightX = abs(rightProbe[0]-bR[0])
    rightY = abs(rightProbe[1]-bR[1])

    print(f'Upper Left Difference: ({leftX},{leftY})')
    print(f'Bottom Right Difference: ({rightX},{rightY})')
    
    return leftX,leftY,rightX,rightY
