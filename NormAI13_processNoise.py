import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

def process_Noise(image_raw,results):
    y_size,x_size = image_raw.shape

    print('  1. defining ROIs')
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
    ROI_snrs = []

    print('  2. constructing figure')
    fig = plt.figure()
    plt.imshow(image_raw,cmap='gray')
    plt.title('Image with ROIs')
    i=0
    ax = plt.gca()
    for ROI in ROIs:
        rect = Rectangle((ROI[0],ROI[2]),ROI[1]-ROI[0],ROI[3]-ROI[2],linewidth=1,edgecolor='red',facecolor='none')
        
        ax.add_patch(rect)
        ax.text(ROI[0], ROI[2], str(i),color='black',fontsize=12,bbox=dict(facecolor='red', alpha=0.5))
        ROI_means.append(np.mean(image_raw[ROI[2]:ROI[3],ROI[0]:ROI[1]]))
        ROI_stds.append(np.std(image_raw[ROI[2]:ROI[3],ROI[0]:ROI[1]]))
        ROI_snrs.append(np.mean(image_raw[ROI[2]:ROI[3],ROI[0]:ROI[1]])/np.std(image_raw[ROI[2]:ROI[3],ROI[0]:ROI[1]]))
        i+=1
    fn = 'image_noise_ROIs.png'
    fig.savefig(fn, bbox_inches = 'tight')
    results.addObject('image_noise_ROIs', fn)
    plt.show()
    plt.close(fig)

    results.addFloat('SNR', np.min(ROI_snrs))

    print('  3. calculating different image characteristics')
    ROI_general_mean = np.mean(ROI_means[0:5])
    ROI_uniformities=[]
    ROI_stds_devs=[]
    for i in range(0,5):
        ROI_uniformities.append(100*((ROI_means[i]-ROI_general_mean)/ROI_general_mean))
        ROI_stds_devs.append(100*(ROI_stds[i]/ROI_means[i]))
    results.addFloat('diff_to_global_mean', np.max(ROI_uniformities)) # <10%
    results.addFloat('sd_perc_ROI', np.max(ROI_stds_devs))   # <5%
            
    return True