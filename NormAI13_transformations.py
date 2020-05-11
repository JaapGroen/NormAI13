from scipy import ndimage
import numpy as np
import cv2
import math


def HorizontalFlip(image, objects):
    rows, cols = image.shape
    image_flipped = np.fliplr(image).copy()
    for object in objects:
        xmin = object['x1']
        xmax = object['x2']
        object['x2'] = cols - xmin
        object['x1'] = cols - xmax
    return image_flipped, objects
        
def VerticalFlip(image, objects):
    rows, cols = image.shape
    image_flipped = np.flipud(image).copy()
    for object in objects:
        ymin = object['y1']
        ymax = object['y2']
        object['y2'] = rows - ymin
        object['y1'] = rows - ymax
    return image_flipped, objects

def Rot90(image, objects, angle):
    rows, cols = image.shape
    if angle==90:
        image_flipped = np.rot90(image,1)
#         image = image_flipped
    elif angle== 270:
        image_flipped = np.rot90(image,3)
#         image = image_flipped
    for object in objects:
        xmin = object['x1']
        xmax = object['x2']
        ymin = object['y1']
        ymax = object['y2']
        if angle==270:
            object['x1'] = rows - ymax
            object['x2'] = rows - ymin
            object['y1'] = xmin
            object['y2'] = xmax
        elif angle==90:
            object['x1'] = ymin
            object['x2'] = ymax
            object['y1'] = cols - xmax
            object['y2'] = cols - xmin
    return image_flipped, objects

def CorrectRotation(image,reshape=True):
    try:
        image_8bit = ((image / np.amax(image)) * 255).astype(np.uint8)
        image_edges = cv2.Canny(image_8bit, 100, 100)
        lines = cv2.HoughLinesP(image_edges, 1, math.pi / 180.0, 50, minLineLength=40, maxLineGap=5)  
        angles = []
        for line in lines:
            [x1, y1, x2, y2] = line[0]
            angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
            angles.append(angle)
        angle = np.median(angles)
        image = ndimage.rotate(image, abs(angle),reshape = reshape, cval=np.mean(image))
        return image,angle
    except:
        return image, None