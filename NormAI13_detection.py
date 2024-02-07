import torch
import torchvision
import pydicom
from pydicom.tag import Tag
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import os

#device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
device = torch.device('cpu')
print('Running prediction on: ',device)

model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
dirpath = os.path.dirname(os.path.realpath(__file__))
saved_model = dirpath+'/models/model_normi13_latest.pth'
# saved_model='models/model_normi13_latest.pth'
model.load_state_dict(torch.load(saved_model, map_location = device))   
model.to(device)
model.eval()

classes={'1':'HighContrast','2':'Linepairs','3':'Noise','4':'LowContrast'}
colors={'1':'red','2':'yellow','3':'blue','4':'green'}
threshold = 0.85

def normi13_detection(dcm_file,results):
    # results = {}
    detection=[]    
    ds = dcm_file     

    print('  1. reading pixelsize')
    try:
        resolution = ds.ImagerPixelSpacing
    except:
        FOV = ds[0x18,0x1149].value
        resolution = [FOV/ds.Columns,FOV/ds.Rows]
    results.addString('Pixelsize', str(resolution[0])+' x '+str(resolution[1]))

    print('  2. prepare image')
    if len(ds.pixel_array.shape)==3:
        print('Multiframe image, getting the middle frame as image.')
        image_detection = ds.pixel_array[round(ds.pixel_array.shape[0]/2),:,:].astype('float32')
    else:
        image_detection = ds.pixel_array.astype('float32')
    image = image_detection
    image_detection = image_detection / np.amax(image_detection)

    image_detection = np.array([image_detection,image_detection,image_detection])
    image_detection = torch.from_numpy(image_detection)
    image_detection = image_detection.to(device)

    print('  3. run prediction model')
    prediction = model([image_detection])

    image_detection.cpu().numpy()
    image_detection = image_detection.cpu().numpy()
    image_detection = np.transpose(image_detection,(1, 2, 0))

    scores = prediction[0]['scores'].detach().numpy()
    scores = scores.tolist()

    print('  4. process outcome')
    for i in range(0,len(scores)):
        if scores[i]>threshold:
            label = str(prediction[0]['labels'][i].cpu().numpy().tolist())
            box = prediction[0]['boxes'][i].detach().numpy()
            rect = Rectangle((box[0],box[1]),box[2]-box[0],box[3]-box[1],
                    linewidth=1,edgecolor=colors[label],facecolor='none')
            ax = plt.gca()
            ax.add_patch(rect)
            ax.text(box[0], box[1], classes[label]+' - '+"%.2f" % scores[i],
                    color='black',fontsize=8,bbox=dict(facecolor=colors[label], alpha=0.5))
            detection.append({'type':label,
                              'score':scores[i],
                              'x1':int(box[0]),
                              'x2':int(box[2]),
                              'y1':int(box[1]),
                              'y2':int(box[3])})

    return image,detection,resolution