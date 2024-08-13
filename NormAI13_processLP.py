import math
import cv2
import scipy
from scipy import ndimage, signal
# import skimage 
import math
import matplotlib.pyplot as plt
import numpy as np
from NormAI13_transformations import *

resolutions = [0.6, 0.7, 0.8, 0.9, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.5, 2.8, 3.1, 3.4, 3.7, 4.0, 4.3, 4.6, 5.0]

def process_LP(image_raw,resolution, results):
    LP_resolutions = [0.6, 0.7, 0.8, 0.9, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.5, 2.8, 3.1, 3.4, 3.7, 4.0, 4.3, 4.6, 5.0]

    print('  1. image processing')
    image_8bit = ((image_raw / np.amax(image_raw)) * 255).astype(np.uint8)
    image_edges = cv2.Canny(image_8bit, 100, 100)

    print('  2. angle detection')
    lines = cv2.HoughLinesP(image_edges, 1, math.pi / 180.0, 100, minLineLength=80, maxLineGap=5)
    if lines is None:
        kernel = np.ones((5, 5), np.uint8)
        image_dilate = cv2.dilate(image_edges, kernel, iterations=2)
        image_erode = cv2.erode(image_dilate, kernel, iterations=1)
        lines = cv2.HoughLinesP(image_erode, 1, math.pi / 180.0, 100, minLineLength=80, maxLineGap=5)
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
        angle = math.degrees(math.atan2(y_max - y_min, x_max - x_min))
        if 40 <= angle <= 50:
            angles.append(angle)
        elif -40 >= angle >= -50:
            angles.append(angle+180)

    angle = np.median([abs(a) for a in angles])
    if 35<angle<55:
        angle = angle+90
    results.addFloat('correction_angle_LP', angle)
    image_rot = ndimage.rotate(image_raw, abs(angle))

    print('  3. build figure')
    fig, axes = plt.subplots(1, 4, figsize=(15, 5))
    axes[0].imshow(image_raw, cmap='gray')
    axes[0].set_title('Original')
    axes[1].imshow(image_8bit, cmap='gray')
    axes[1].set_title('8bit')
    axes[2].imshow(image_edges, cmap='gray')
    axes[2].set_title('Edges')
    axes[3].imshow(image_rot, cmap='gray')
    axes[3].set_title('Rotated')
    fn = 'image_LP_processing.png'
    plt.savefig(fn, bbox_inches = 'tight')
    results.addObject('image_LP_processing', fn)
    plt.show()
    plt.close(fig)

    print('  4. cut out bars')
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

    print('  5. build figure of the bars')   
    fig = plt.figure()
    plt.subplot(1,2,1)
    plt.imshow(image_1,cmap='gray')
    plt.title('bars 1')
    plt.subplot(1,2,2)
    plt.imshow(image_2,cmap='gray')
    plt.title('bars 2')

    fn = 'image_LP_bars.png'
    fig.savefig(fn, bbox_inches = 'tight')
    results.addObject('image_LP_bars', fn)
    plt.show()
    plt.close(fig)

    print('  6. make profiles through the bars and detect peaks')
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

    print('  7. plot the profiles')
    fig = plt.figure()
    plt.subplot(2,1,1)
    plt.plot(profile_lb)
    plt.scatter(peaks_lb[0],peaks_lb[1]['peak_heights'],marker='o',c='green')
    plt.title('Large bars profile')
    plt.subplot(2,1,2)
    plt.plot(profile_sb)
    plt.scatter(peaks_sb[0],peaks_sb[1]['peak_heights'],marker='o',c='green')
    plt.title('Small bars profile')

    fn = 'plot_LP_bars.png'
    fig.savefig(fn, bbox_inches = 'tight')
    results.addObject('plot_LP_bars', fn)
    plt.show()
    plt.close(fig)

    print('  8. count the visible lines')
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

    results.addFloat('visible_resolution', LP_resolutions[detected_groups-1])
    
    return True
