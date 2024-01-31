from scipy import signal
import cv2
import math
import matplotlib.pyplot as plt
from NormAI13_transformations import *

def process_HC(image_raw):
    results = {}
    
    # show image we are working with
    fig = plt.figure()
    plt.imshow(image_raw, cmap='gray')
    plt.title('Input image')
    results['raw input'] = fig
#     plt.close()
    
    image_rot,angle = CorrectRotation(image_raw,reshape = False)
    results['correction angle raw image'] = angle
    
    if 42<abs(angle)<48:
        print('Angle cannot be {angle:.2f} degrees, revert the correction.'.format(angle=angle))
        image_rot = image_raw
        results['correction angle raw image'] = 0
    
    #use vert profile to crop the image in the vertical direction
    vert_profile = np.mean(image_rot,axis=1)
    vert_profile_mean = np.mean(vert_profile)
    index_start = 0
    index_end = len(vert_profile)
    for i in range(0,len(vert_profile)-10):
        t = vert_profile[i:i+9]
        if (np.mean(t) < vert_profile_mean*1.05) and (np.mean(t) > vert_profile_mean*0.95):
            index_start = i+9
            break
    for i in range(0,len(vert_profile)-10):
        t = vert_profile[i:i+9]
        if (np.mean(t) < vert_profile_mean*1.05) and (np.mean(t) > vert_profile_mean*0.95):
            index_end = i
    image_crop = image_rot[index_start:index_end,:]  

    # use horizontal profile to crop in horizontal direction
    hor_profile = np.mean(image_crop,axis=0)      
    hor_profile_mean = np.mean(hor_profile)
    index_start = 0
    index_end = len(hor_profile)  
    for i in range(0,len(vert_profile)-5):
        t = hor_profile[i:i+4]
        if (np.std(t) < 2):
            index_start = i+4
            break
    for i in range(0,len(hor_profile)-5):
        t = hor_profile[i:i+5]
        if (np.std(t) < 2):
            index_end = i
    image_crop2 = image_crop[:,index_start:index_end] 
    
    left_part = image_crop2[0:image_crop2.shape[0],0:int(image_crop2.shape[1]/2)]
    right_part = image_crop2[0:image_crop2.shape[0],int(image_crop2.shape[1]/2):image_crop2.shape[1]]
    
    #get the high values to the right otherwise the peaks are negative
    if np.mean(left_part)>np.mean(right_part):
        image_corr = np.fliplr(image_crop2)
    else:
        image_corr = image_crop2
    
    # show image we are working with
    fig = plt.figure()
    plt.imshow(image_corr, cmap='gray')
    plt.title('corrected image')
    results['corrected_image'] = fig
       
    # use new horizontal profile to find the different parts, peak height can be used as lower detection limit.
    hor_profile = np.mean(image_corr,axis=0)
    diff = np.diff(hor_profile)
    
    updating = True
    height = 15
    distance = round(len(diff)/10)
    print('dist',distance)
    while updating:
        peaks = signal.find_peaks(diff,height = height,distance = distance)
        if len(peaks[0])>6:
            height = height+1
        else:
            updating = False
            results['height'] = height
            
    fig = plt.figure()
    plt.plot(hor_profile)
    plt.plot(diff)
    plt.scatter(peaks[0],peaks[1]['peak_heights'],marker = 'o')
    plt.title('Profile and peakdetection')
    results['profile image'] = fig
#     plt.close()
    
    contrasts = []
    for i in range(0,len(peaks[0])):
        if len(contrasts) == 0:   #first one
            contrasts.append(np.mean(hor_profile[5:peaks[0][i]-5]))
        else:
            contrasts.append(np.mean(hor_profile[peaks[0][i-1]+5:peaks[0][i]]))
    contrasts.append(np.mean(hor_profile[peaks[0][-1]+5:len(hor_profile)-5]))
    
#     print('Number of visible elements:',len(contrasts))
    results['num_elem'] = len(contrasts)
    
    true_contrasts = [0, 0.3, 0.65, 1, 1.4, 1.85, 2.30]
    contrasts = contrasts - np.min(contrasts)
    scale = np.max(contrasts) / true_contrasts[-1]
    phantom_contrasts = [c * scale for c in true_contrasts]
    
    fig = plt.figure()
    plt.plot(contrasts, 'b')
    plt.plot(phantom_contrasts, 'r')
    plt.legend(['Measured','True'])
    plt.title('Contrast plot')
    plt.grid(True)
    results['HighContrast_plot'] = fig  
    
    return results