import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import math
from scipy.signal import find_peaks

def fitEllipse(x,y):
    x=x[:,None]
    y=y[:,None]

    D=np.hstack([x*x,x*y,y*y,x,y,np.ones(x.shape)])
    S=np.dot(D.T,D)
    C=np.zeros([6,6])
    C[0,2]=C[2,0]=2
    C[1,1]=-1
    E,V=np.linalg.eig(np.dot(np.linalg.inv(S),C))
    n=np.argmax(E)
    a=V[:,n]
    return a

def BilinearInterpolation(row,col,image):
    c1 = math.floor(col)
    c2 = math.ceil(col)
    r1 = math.floor(row)
    r2 = math.ceil(row)
    m11 = image[r1,c1]
    m12 = image[r1,c2]
    m21 = image[r2,c1]
    m22 = image[r2,c2]
    value = np.interp(col, [c1,c2], [np.interp(row, [r1, r2], [m11, m21]),np.interp(row, [r1, r2], [m12, m22])])
    return value

def ellipse_center(a):
    b,c,d,f,g,a = a[1]/2, a[2], a[3]/2, a[4]/2, a[5], a[0]
    num = b*b-a*c
    x0=(c*d-b*f)/num
    y0=(a*f-b*d)/num
    return np.array([x0,y0])

def ellipse_axis_length(a):
    b,c,d,f,g,a = a[1]/2, a[2], a[3]/2, a[4]/2, a[5], a[0]
    up = 2*(a*f*f+c*d*d+g*b*b-2*b*d*f-a*c*g)
    down1=(b*b-a*c)*( (c-a)*np.sqrt(1+4*b*b/((a-c)*(a-c)))-(c+a))
    down2=(b*b-a*c)*( (a-c)*np.sqrt(1+4*b*b/((a-c)*(a-c)))-(c+a))
    res1=np.sqrt(up/down1)
    res2=np.sqrt(up/down2)
    return np.array([res1, res2])

def ellipse_angle_of_rotation2(a):
    b,c,d,f,g,a = a[1]/2, a[2], a[3]/2, a[4]/2, a[5], a[0]
    if b == 0:
        if a > c:
            return 0
        else:
            return np.pi/2
    else:
        if a > c:
            return np.arctan(2*b/(a-c))/2
        else:
            return np.pi/2 + np.arctan(2*b/(a-c))/2
            
def find_circle_center(image,resolution):  
    fig, axes = plt.subplots(1, 5, figsize=(15, 5))
    axes[0].imshow(image, cmap='gray')
    axes[0].set_title('Original')
    
    threshold = np.mean(image)
    mask = image < threshold
    mask = mask.astype('uint8')
    axes[1].imshow(mask, cmap='gray')
    axes[1].set_title('Mask')

    # Find connected components in the mask
    _, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)

    # Get label of the background (largest component)
    background_label = np.argmax(stats[1:, cv2.CC_STAT_AREA]) + 1 if len(stats) > 1 else -1

    # Remove small connected components
    min_area_threshold_mm2 = 5  # Minimum area in mm²
    min_area_threshold_pixels = int(min_area_threshold_mm2 / (resolution[0] * resolution[0]))

    # Find connected components in the modified mask
    _, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)

    # Remove small areas and those touching the border, except the largest component
    for label, stat in enumerate(stats):
        if label != background_label and (stat[4] < min_area_threshold_pixels or
                                          np.any(np.isin(np.where(labels == label)[0], [0, mask.shape[0]-1])) or
                                          np.any(np.isin(np.where(labels == label)[1], [0, mask.shape[1]-1]))):
            mask[labels == label] = 0

    axes[2].imshow(mask, cmap='gray')
    axes[2].set_title('Filtered')

    sobel_x = cv2.Sobel(mask, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(mask, cv2.CV_64F, 0, 1, ksize=3)
    image_edges = np.sqrt(sobel_x**2 + sobel_y**2)
    axes[3].imshow(image_edges, cmap='gray')
    axes[3].set_title('Edges')

    rows, cols = image_edges.shape

    estimate_x = rows//2
    estimate_y = cols//2
    
    max_estimate = int(7 / resolution[0])
    while max_estimate>rows or max_estimate>cols: #if object is close to the edge.
        max_estimate = max_estimate-1   
    min_estimate= 0
    skip_angle = 0
    
    # we get the last part of the profile along an angle and look for the edge coordinates (angle and radius)
    edge_x=[]
    edge_y=[]
    angles = np.arange(0+math.radians(skip_angle)-np.pi/2,2*np.pi-math.radians(skip_angle)-np.pi/2, 0.01)
    # for angle in np.arange(0+math.radians(skip_angle)-np.pi/2,2*np.pi-math.radians(skip_angle)-np.pi/2, 0.01):
    for angle in angles:
        v=[]
        r=[]
        for rsub in range(min_estimate,max_estimate,1):
            x=rsub * math.sin((angle)) + estimate_x
            y=rsub * math.cos((angle)) + estimate_y
            try:
                v.append(BilinearInterpolation(int(x),int(y),image_edges))     #subpixel resolution
                r.append(rsub)
            except:
                pass
        pixsum=0
        rowsum=0                
        for i in range(0,len(v),1):
            if v[i] is not None:
                pixsum=pixsum+v[i]
                rowsum=rowsum+v[i]*r[i]
        if pixsum>0:
            r_edge=rowsum/pixsum
        else:
            r_edge=None
        
        if r_edge:
            xt=r_edge * math.sin((angle)) + estimate_x
            yt=r_edge * math.cos((angle)) + estimate_y
            edge_x.append(xt)
            edge_y.append(yt)

    axes[4].imshow(image, cmap='gray')
    axes[4].set_title('ROI')
    
    # fit the found edge points to an ellipse
    edge_x = np.asarray(edge_x)
    edge_y = np.asarray(edge_y)
    a = fitEllipse(edge_x,edge_y)
    center = ellipse_center(a)
    phi = ellipse_angle_of_rotation2(a)
    axes_ellips = ellipse_axis_length(a)
    radius = np.mean(axes_ellips)

    ellipse = plt.Circle((center[1], center[0]), radius, color='r', fill=False)
    axes[4].add_patch(ellipse)
    plt.show()
    plt.close(fig)
    
    return center, phi, axes_ellips, fig
    
def calculate_contrast(image, center, resolution, annulus_outer_radius, annulus_inner_radius):
    image = np.ascontiguousarray(image)
    
    # Create a circular mask for the region of interest (ROI)
    circle_mask = np.zeros_like(image, dtype=np.uint8)
    cv2.circle(circle_mask, (int(center[0]), int(center[1])),
               int(3 / resolution[0]), 1, -1)  # ROI with radius of 3 mm

    # Calculate the mean pixel value within the circular ROI
    mean_value = np.mean(image[circle_mask == 1])

    # Create an annulus mask
    annulus_mask = np.zeros_like(image, dtype=np.uint8)
    cv2.circle(annulus_mask, (int(center[0]), int(center[1])),
               int(annulus_outer_radius / resolution[0]), 1, -1)
    cv2.circle(annulus_mask, (int(center[0]), int(center[1])),
               int(annulus_inner_radius / resolution[0]), 0, -1)

    # Ensure circular ROI pixels are excluded from the annulus calculation
    annulus_mask[circle_mask == 1] = 0

    # Calculate the mean pixel value within the annulus
    mean_annulus = np.mean(image[annulus_mask == 1])

    # Calculate contrast
    contrast = 100 * (abs(mean_annulus - mean_value) / mean_annulus)

    # Calculate the Contrast-to-Noise Ratio (CNR)
    noise_std = np.std(image[annulus_mask == 0])
    cnr = abs(mean_value - mean_annulus) / noise_std

    return mean_value, mean_annulus, contrast, cnr

def get_roi_mean_values(image):
    roi_size = (30, 30)  # in pixels
    rois = [
        image[:roi_size[0], :roi_size[1]],          # Top-left corner
        image[-roi_size[0]:, :roi_size[1]],         # Top-right corner
        image[:roi_size[0], -roi_size[1]:],         # Bottom-left corner
        image[-roi_size[0]:, -roi_size[1]:],        # Bottom-right corner
    ]
    mean_values = [np.mean(roi) for roi in rois]
    return mean_values

def remove_gradient(image, resolution):
    original_roi_means = get_roi_mean_values(image)
    correction_values = original_roi_means - np.mean(image)
    gradient_image = np.zeros_like(image)
    rows, cols = image.shape
    for i in range(rows):
        for j in range(cols):
            gradient_image[i, j] = i * correction_values[0] / rows + (rows - i) * correction_values[2] / rows + \
                                   j * correction_values[3] / cols + (cols - j) * correction_values[1] / cols
    corrected_image = image - gradient_image
    return corrected_image

def process_LowContrast(image, resolution,results):
    LC_centers = []

    # Sizes in mm for drawing and calculating contrast
    object_radius = 5
    annulus_inner_radius = 7
    annulus_outer_radius = 9

    print('  1. image processing')
    # first some image processing
    normalized_image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)  # Normalize the image to have pixel values in the range [0, 255]
    denoised_image = cv2.GaussianBlur(normalized_image, (25, 25), 0)  # Increase denoising by applying Gaussian blur with larger kernel size

    # Profile of the image along the horizontal axis using mean
    profile = np.mean(denoised_image, axis=0)

    print('  2. estimating location of the first object')
    # Find the locations of negative peaks
    min_peak_width = 20  # Adjust this threshold as needed
    negative_peaks, _ = find_peaks(-profile, width=min_peak_width)

    # Print the x-coordinate of the first negative peak
    if negative_peaks.size > 0:
        first_object_x = negative_peaks[0]
        print(f'  The x-coordinate of the first object: {first_object_x}')    
    else:
        print('  No negative peaks detected.')

    print('  3. cutting out the first object and processing')
    # Cut out a larger portion of the denoised image around the first object with a diameter of 10 mm and larger margin
    object_diameter_mm = 10  # Diameter for the cut-out
    margin_mm = 4  # Increased margin

    # Cut out a portion of the image around the specified x-coordinate with a larger margin
    pixels_per_mm = 1 / resolution[0]
    diameter_pixels = int(object_diameter_mm * pixels_per_mm)
    margin_pixels = int(margin_mm * pixels_per_mm)

    x_start = max(0, first_object_x - diameter_pixels // 2 - margin_pixels)
    x_end = min(denoised_image.shape[1], first_object_x + diameter_pixels // 2 + margin_pixels)
    portion = denoised_image[:, x_start:x_end]

    portion = remove_gradient(portion, resolution)

    print('  4. subpixel center detection of first object')
    # Subpixel object detection
    detected_center, detected_angle, detected_axes,fig = find_circle_center(portion, resolution)
    circle_center = (detected_center[1] + x_start, detected_center[0])
    print('  first object center:',circle_center)
    LC_centers.append(circle_center)

    fn = 'image_first_LC_detection.png'
    fig.savefig(fn, bbox_inches = 'tight')
    results.addObject('image_first_LC_detection', fn)
    plt.show()
    plt.close(fig)

    print('  5. cutting out the second object and processing')
    # Estimate the second object
    second_object_x = int(20 / resolution[0] + circle_center[0])
    x_start = max(0, int(second_object_x) - diameter_pixels // 2 - margin_pixels)
    x_end = min(image.shape[1], int(second_object_x) + diameter_pixels // 2 + margin_pixels)
    portion = denoised_image[:, x_start:x_end]

    portion = remove_gradient(portion, resolution)

    print('  6. subpixel center detection of second object')
    # Subpixel object detection
    detected_center, detected_angle, detected_axes, fig = find_circle_center(portion, resolution)
    circle_center = (detected_center[1] + x_start, detected_center[0])
    print('  second object center:',circle_center)
    LC_centers.append(circle_center)

    fn = 'image_second_LC_detection.png'
    fig.savefig(fn, bbox_inches = 'tight')
    results.addObject('image_second_LC_detection', fn)
    plt.show()
    plt.close(fig)
    
    # Calculate offsets
    x_distance = LC_centers[1][0] - LC_centers[0][0]
    y_distance = LC_centers[1][1] - LC_centers[0][1]

    print('  7. defining other ROIs')
    # Objects positioned in a straight line, thus add 4 remaining circles based upon distances
    for x in range(0,4):
        LC_centers.append([LC_centers[-1][0]+x_distance,LC_centers[-1][1]+y_distance])

    print('  8. calculating contrasts')
    contrasts = []
    cnrs = []
    for center in LC_centers:
        try:
            mean_value,annulus_value,contrast,cnr = calculate_contrast(image,center,resolution,annulus_outer_radius,annulus_inner_radius)
            contrasts.append(contrast)
            cnrs.append(cnr)
        except Exception as e:
            print('Error calculating contrast and cnr:',e)
            pass

    print('  9. building figures')
    # Build the image
    fig = plt.figure()
    plt.imshow(image, cmap='gray')
    plt.title('LC objects with ROIs')
    ax = plt.gca()
    for center in LC_centers:
        circle = plt.Circle((center[0], center[1]), 3/resolution[0], edgecolor='red', facecolor='none')    #object has radius of 5mm
        ax.add_patch(circle)
        annulus_outer = plt.Circle((center[0], center[1]),
                         annulus_outer_radius/resolution[0], edgecolor='blue', facecolor='none', linestyle='dashed')
        ax.add_patch(annulus_outer)
        annulus_inner = plt.Circle((center[0], center[1]),
                         annulus_inner_radius/resolution[0], edgecolor='blue', facecolor='none', linestyle='dashed')
        ax.add_patch(annulus_inner)
    fn = 'image_LC_ROIs.png'
    fig.savefig(fn, bbox_inches = 'tight')
    results.addObject('image_LC_ROIs', fn)
    plt.show()
    plt.close(fig)

    print('  10. building plots')
    # contrastplot
    true_contrasts = [5.6, 4.0, 2.8, 2.0, 1.2, 0.8]
    fig = plt.figure()
    plt.plot(contrasts,'b')
    plt.plot(true_contrasts,'r')
    plt.title('Low Contrast plot')
    plt.legend(['Measured','True'])
    plt.grid(True)
    fn = 'plot_LC.png'
    fig.savefig(fn, bbox_inches = 'tight')
    results.addObject('plot_LC', fn)
    plt.show()
    plt.close(fig)
    
    fig = plt.figure()
    plt.plot(cnrs,'b')
    plt.title('CNR plot')
    plt.grid(True)
    fn = 'plot_CNR.png'
    fig.savefig(fn, bbox_inches = 'tight')
    results.addObject('plot_CNR', fn)
    plt.show()
    plt.close(fig)
    
    return True