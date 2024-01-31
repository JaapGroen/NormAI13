import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

def process_Noise(image_raw):
    results = {}
    
    fig = plt.figure()
    plt.imshow(image_raw,cmap='gray')
    plt.title('raw image with ROIs')
    
    y_size,x_size = image_raw.shape
    
    ROIs=[
        [int((1/7)*x_size),int((2/7)*x_size),int((1/7)*y_size),int((2/7)*y_size)],
        [int((5/7)*x_size),int((6/7)*x_size),int((1/7)*y_size),int((2/7)*y_size)],
        [int((5/7)*x_size),int((6/7)*x_size),int((5/7)*y_size),int((6/7)*y_size)],
        [int((1/7)*x_size),int((2/7)*x_size),int((5/7)*y_size),int((6/7)*y_size)],
        [int((3/7)*x_size),int((4/7)*x_size),int((3/7)*y_size),int((4/7)*y_size)],
        [int((5/7)*x_size),int((6/7)*x_size),int((3/7)*y_size),int((4/7)*y_size)]
    ]
    
    ROI_means = []
    ROI_stds = []
    
    i=0
    for ROI in ROIs:
        rect = Rectangle((ROI[0],ROI[2]),ROI[1]-ROI[0],ROI[3]-ROI[2],linewidth=1,edgecolor='red',facecolor='none')
        ax = plt.gca()
        ax.add_patch(rect)
        ax.text(ROI[0], ROI[2], str(i),color='black',fontsize=12,bbox=dict(facecolor='red', alpha=0.5))
        ROI_means.append(np.mean(image_raw[ROI[2]:ROI[3],ROI[0]:ROI[1]]))
        ROI_stds.append(np.std(image_raw[ROI[2]:ROI[3],ROI[0]:ROI[1]]))
        i+=1
        
    plt.show()
    plt.close()
    
#     results['uniformity_correction'] = np.mean(image_raw,axis=0))
        
    results['Uniformity image'] = fig
    
    maxdev = abs(ROI_means[0]-ROI_means[4])
    for i in range(0,4):
        maxdev = max(maxdev,abs(ROI_means[i]-ROI_means[4]))
    
    unif_pct = 100.*maxdev/ROI_means[4]
    snr_hol = ROI_means[5]/ROI_stds[5]
    results['maximum deviation'] = maxdev
    results['uniformity'] = unif_pct
    results['snr'] = snr_hol
    
    ROI_general_mean = np.mean(ROI_means[0:5])
    
    ROI_uniformities=[]
    ROI_stds_devs=[]
    for i in range(0,5):
        ROI_uniformities.append(100*((ROI_means[i]-ROI_general_mean)/ROI_general_mean))
        ROI_stds_devs.append(100*(ROI_stds[i]/ROI_means[i]))
    results['uniformity_A'] = np.max(ROI_uniformities)
    results['uniformity_B'] = np.max(ROI_stds_devs)
    
    
    return results