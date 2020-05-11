import math
import cv2
import scipy
from scipy import ndimage, signal
import skimage 
import math
import matplotlib.pyplot as plt
import numpy as np
from NormAI13_transformations import *

resolutions = [0.6, 0.7, 0.8, 0.9, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.5, 2.8, 3.1, 3.4, 3.7, 4.0, 4.3, 4.6, 5.0]

def process_LP(image_raw,resolution):
    results = {}
    
    # show image we are working with
    fig = plt.figure()
    plt.imshow(image_raw,cmap='gray')
    plt.title('raw image with detected lines')
    
    
    # detect to angle of rotation
    image_8bit = ((image_raw / np.amax(image_raw)) * 255).astype(np.uint8)
    image_edges = cv2.Canny(image_8bit, 100, 100) 
    lines = cv2.HoughLinesP(image_edges, 1, math.pi / 180.0, 100, minLineLength=80, maxLineGap=5)  
    angles = []
    x_min, y_min = image_raw.shape
    x_max = 0
    y_max = 0
    for line in lines:
        [x1, y1, x2, y2] = line[0]
        if np.min([x1,x2])<x_min:
            x_min = np.min([x1,x2])
        if np.min([y1,y2])<y_min:
            y_min = np.min([y1,y2])
        if np.max([x1,x2])>x_max:
            x_max = np.max([x1,x2])
        if np.max([y1,y2])>y_max:
            y_max = np.max([y1,y2])
        angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
        if 40 <= angle <= 50:
            angles.append(angle)
            plt.plot([x1,x2],[y1,y2])
        elif -40 >= angle >= -50:
            angles.append(angle+180)
            plt.plot([x1,x2],[y1,y2])
    results['raw image'] = fig
    
    angle = np.median([abs(a) for a in angles])
    results['correction angle'] = angle

    # rotate the image by detected angle
    image_rot = ndimage.rotate(image_raw, abs(angle))
    
    # show image we are working with
    fig = plt.figure()
    plt.imshow(image_rot,cmap='gray')
    plt.title('rotated raw image')
    results['corrected image'] = fig
    
       
    # calculate to number of pixels needed for the bars section based on the pixelspacing
    lines_width = int(43/resolution[0])
    lines_height = int(34/resolution[1])
    
    # cut out the bars section
    image_corr = image_rot[int((image_rot.shape[0]/2)-(lines_width/2))+10:int((image_rot.shape[0]/2)+(lines_width/2))+5,
                           int((image_rot.shape[1]/2)-(lines_height/2))+10:int((image_rot.shape[1]/2)+(lines_height/2))+5]        
    
    # calculate to number of pixels needed for a single bar section based on the pixelspacing
    bars_width = int(14/resolution[0])
    offset = int((image_corr.shape[1]-2*bars_width)/4)
    
    # split into two images containing the two bars section
    image_1 = image_corr[0:image_corr.shape[0],offset:int(image_corr.shape[1]/2)-offset]
    image_2 = image_corr[0:image_corr.shape[0],int(image_corr.shape[1]/2)+offset:image_corr.shape[1]-offset]
    
    # correct the bar images for rotation and crop them
    image_1,angle_1 = CorrectRotation(image_1,reshape = False)
    results['bars 1 angle correction'] = angle_1
    image_2,angle_2 = CorrectRotation(image_2,reshape = False)
    results['bars 2 angle correction'] = angle_2
        
    fig = plt.figure()
    plt.subplot(1,2,1)
    plt.imshow(image_1,cmap='gray')
    plt.title('bars 1')
    plt.subplot(1,2,2)
    plt.imshow(image_2,cmap='gray')
    plt.title('bars 2')
    results['image_bars'] = fig
    
    # make a profile, correct the gradient and detect the peaks
    profile_1 = np.mean(image_1,axis=1)
    profile_2 = np.mean(image_2,axis=1)
    fit_1 = np.polyfit(range(0,len(profile_1)),profile_1,1)
    fit_2 = np.polyfit(range(0,len(profile_2)),profile_2,1)      
    profile_1 = profile_1 - range(0,len(profile_1))*fit_1[0]+fit_1[1]
    profile_2 = profile_2 - range(0,len(profile_2))*fit_2[0]+fit_2[1]
    
    # peak detection
    height = (np.amax(profile_1)-np.amin(profile_1))*0.5+np.amin(profile_1)
    peaks_1 = scipy.signal.find_peaks(profile_1,height = height,distance = 2)
        
    height = (np.amax(profile_2)-np.amin(profile_2))*0.5+np.amin(profile_2)
    peaks_2 = signal.find_peaks(profile_2,height = height,distance = 2)

    
    if len(peaks_1) > len(peaks_2):
        profile_lb = profile_1
        profile_sb = profile_2
        peaks_lb = peaks_1
        peaks_sb = peaks_2
    else:
        profile_lb = profile_2
        profile_sb = profile_1
        peaks_lb = peaks_2
        peaks_sb = peaks_1
    
    fig = plt.figure()
    plt.subplot(2,1,1)
    plt.plot(profile_lb)
    plt.scatter(peaks_lb[0],peaks_lb[1]['peak_heights'],marker='o',c='green')
    plt.title('Large bars profile')
    plt.subplot(2,1,2)
    plt.plot(profile_sb)
    plt.scatter(peaks_sb[0],peaks_sb[1]['peak_heights'],marker='o',c='green')
    plt.title('Small bars profile')
    results['bar_profiles'] = fig
    results['lb peaks'] = len(peaks_lb[0])
    results['sb peaks'] = len(peaks_lb[0])
    
    if len(peaks_lb[0]) < 24:
        first_distance = peaks_lb[0][1]-peaks_lb[0][0]
        previous_distance = first_distance
        groups = [[peaks_lb[0][0],first_distance]]
        for i in range(2,len(peaks_lb[0])):
            distance_to_previous = peaks_lb[0][i]-peaks_lb[0][i-1]
            if previous_distance-2 <= distance_to_previous <= previous_distance+2:
                groups[len(groups)-1].append(distance_to_previous)
                previous_distance = distance_to_previous
            else:
                groups.append([distance_to_previous])
        detected_groups = len([x for x in groups if len(x)==3])
    else:
        first_distance = peaks_sb[0][1]-peaks_sb[0][0]
        previous_distance = first_distance
        groups = [[peaks_sb[0][0],first_distance]]
        for i in range(2,len(peaks_sb[0])):
            distance_to_previous = peaks_sb[0][i]-peaks_sb[0][i-1]
            if previous_distance-2 <= distance_to_previous <= previous_distance+2:
                groups[len(groups)-1].append(distance_to_previous)
                previous_distance = distance_to_previous
            else:
                groups.append([distance_to_previous])
        detected_groups = len([x for x in groups if len(x)==3])
        detected_groups = detected_groups + 8 
    results['resolution visible'] = resolutions[detected_groups-1]
    
    return results