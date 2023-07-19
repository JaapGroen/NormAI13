import cv2
import scipy.ndimage as ndimage
from matplotlib.patches import Circle, Rectangle
import matplotlib.pyplot as plt
from NormAI13_transformations import *

# find a cirle with a moving mask and low mean value and low std
def find_LC(image,radius):
    best_fit = {'mean':99999,'std':99999,'x':0,'y':0}
    height, width = image.shape
    x, y = np.meshgrid(np.arange(width), np.arange(height))
    
    for x_center in range(radius,width-radius):
        for y_center in range(radius,height-radius):
            d2 = (x - x_center)**2 + (y - y_center)**2
            mask = d2 < radius**2
            ROImean = np.mean(image[mask])
            ROIstd = np.std(image[mask])
            if ROImean < best_fit['mean'] and ROIstd < best_fit['std']*1.05:
                best_fit['mean'] = ROImean
                best_fit['std'] = ROIstd
                best_fit['x'] = x_center
                best_fit['y'] = y_center
    return best_fit

# function for getting circular ROI statistics
def measure_ROI(image,ROI_centerX,ROI_centerY,ROI_radius):
    H, W = image.shape
    x, y = np.meshgrid(np.arange(W), np.arange(H))
    d2 = (x - ROI_centerX)**2 + (y - ROI_centerY)**2
    mask = d2 < ROI_radius**2
    ROImean=np.mean(image[mask])
    ROIstd=np.std(image[mask])
    ROIsize=np.sum(mask)
    return ROImean,ROIstd,ROIsize

# main function
def process_LowContrast(image_raw,resolution):
    results = {}
    
    # object diameter = 10 mm
    object_radius = int(5 / resolution[0]) 
    
    # crop the image by removing all rows with a 0 in it.
    for i in range(image_raw.shape[0]-1,0,-1):
        if np.min(image_raw[i,:]) != 0:
            cutoff_y = i
            break
    image_crop = image_raw[0:cutoff_y-1,0:image_raw.shape[1]]

    # correct image for offset
    corr = np.mean(image_crop,axis=0)
    image_crop_corr = image_crop - corr
    
    # look for the 2 first circles, starting with the biggest contrast
    found_circles = []
    x_border = 0
    while len(found_circles)<2:   # we stop when we have 2 objects
        image_sub = image_crop_corr[:,x_border+object_radius:image_crop_corr.shape[1]]
        best_fit = find_LC(image_sub,object_radius) 
        best_fit['x'] = best_fit['x'] + x_border + object_radius
        x_border = best_fit['x'] + object_radius   # we cut off the part with the just found circle
        found_circles.append(best_fit)
    results['found circles'] = len(found_circles)

    # plot the found circles in red
    fig = plt.figure()
    plt.imshow(image_crop,cmap='gray')
    plt.title('Cropped image with ROIs')
    for circle in found_circles:
        circ = Circle((circle['x'],circle['y']),object_radius,linewidth=1,edgecolor='red',facecolor='none')
        ax = plt.gca()
        ax.add_patch(circ)
        
    # distances between 1 and 2
    x_distance = found_circles[1]['x'] - found_circles[0]['x']
    y_distance = found_circles[1]['y'] - found_circles[0]['y']
    
    # add the first 2 circles
    circles = []
    circles.append(found_circles[0])
    circles.append(found_circles[1])
    
    # circles positioned in a straight line, thus add 4 remaining circles based upon distances
    for x in range(0,4):
        circles.append({'x':circles[-1]['x']+x_distance,'y':circles[-1]['y']+y_distance})
 
    # properties of the background part
    background = {}
    background['x'] = 0
    background['y'] = 0
    background['width'] = image_crop.shape[1]
    background['height'] = np.min([circles[0]['y']-object_radius-5,circles[5]['y']-object_radius-5])  # the line might be angled
    image_background = image_crop[background['x']:background['x']+background['width'],
                                  background['y']:background['y']+background['height']]
    mean_background = np.mean(image_background)
 
    # reduce the radius for measurement in case of sub-optimal position 
    object_radius_measure = int(4 / resolution[0])

    # add the circles to the plot in blue
    
    for circle in circles:
        circ = Circle((circle['x'],circle['y']),object_radius_measure,linewidth=1,edgecolor='blue',facecolor='none')
        ax = plt.gca()
        ax.add_patch(circ)
    # add the background to the plot in yellow
    rect = Rectangle((background['x'],background['y']),background['width'],background['height'],linewidth=1,edgecolor='yellow',facecolor='none')
    ax = plt.gca()
    ax.add_patch(rect)
    results['image with ROIs'] = fig  

    # calculate contrasts
    true_contrasts = [5.6, 4.0, 2.8, 2.0, 1.2, 0.8]
    contrasts = []
    for circle in circles:
        ROImean,ROIstd,ROIsize = measure_ROI(image_crop,circle['x'],circle['y'],object_radius_measure)
        contrasts.append((((mean_background-ROImean)/mean_background)*100))

    # contrastplot
    fig = plt.figure()
    plt.plot(contrasts,'b')
    plt.plot(true_contrasts,'r')
    plt.title('Low Contrast plot')
    plt.legend(['Measured','True'])
    plt.grid(True)
    results['LowContrast_plot'] = fig
    
    return results
