import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
from NormAI13_transformations import *

def process_detection(image,detection,results):
    transformations = []
    
    x_size, y_size = image.shape 
    
    mask = np.zeros(image.shape)
    rico = image.shape[1] / image.shape[0]
    for x in range(0,image.shape[0]):
        for y in range(0,image.shape[1]):
            if y<rico*x and y<image.shape[1]-x*rico:
                mask[x,y] = 1
            elif y>rico*x and y<image.shape[1]-x*rico:
                mask[x,y] = 2
            elif y>rico*x and y>image.shape[1]-x*rico:
                mask[x,y] = 3
   
    objects = []

    print('  1. determine the postion of detected objects')
    for object in detection:
        x1 = object['x1']
        x2 = object['x2']
        y1 = object['y1']
        y2 = object['y2']

        center_x = int((x1+x2)/2)
        center_y = int((y1+y2)/2)
        
        if mask[center_y,center_x]==1:
            object['position']='west'
        elif mask[center_y,center_x]==2:
            object['position']='north'
        elif mask[center_y,center_x]==3:
            object['position']='east'
        else:
             object['position']='south'
        objects.append(object)
    
    for object in detection:
        if object['type']=='1':
            HC_object = object
        elif object['type']=='2':
            LP_object = object

    print('  2. figuring out the orientation of the phantom and applying transformations')
    if HC_object['position'] == 'north':
        if LP_object['position'] == 'west':
            transformations.append('No flips/rotations required')
        elif LP_object['position'] == 'east':
            transformations.append('Flipping the image horizontally')
            image,detection = HorizontalFlip(image, objects)
    elif HC_object['position'] == 'east':
        if LP_object['position'] == 'north':
            transformations.append('Rotating the image 90 degrees')
            image,detection = Rot90(image, objects, 90)
        elif LP_object['position'] == 'south':
            transformations.append('Flipping the image vertically')
            image,detection = VerticalFlip(image, objects)
            transformations.append('Rotating the image 90 degrees')
            image,detection = Rot90(image, objects, 90)     
    elif HC_object['position'] == 'south':
        if LP_object['position'] == 'west':
            transformations.append('Flipping the image vertically')
            image,detection = VerticalFlip(image, objects)
        elif LP_object['position'] == 'east':
            transformations.append('Flipping the image vertically')
            image,detection = VerticalFlip(image, objects)
            transformations.append('Flipping the image horizontally')
            image,detection = HorizontalFlip(image, objects)            
    elif HC_object['position'] == 'west':
        if LP_object['position'] == 'north':
            transformations.append('Flipping the image vertically')
            image,detection = VerticalFlip(image, objects)
            transformations.append('Rotating the image 270 degrees')
            image,detection = Rot90(image, objects, 270)
        elif LP_object['position'] == 'south':
            transformations.append('Rotating the image 270 degrees')
            image,detection = Rot90(image, objects, 270)

    print('  3. place other objects')
    HC_object['position'] = 'north'
    LP_object['position'] = 'west'
    Noise_object=({'type':'3','position':'east'})
    Noise_object['x1'] = max([LP_object['x1'],LP_object['x2']])+int(x_size*0.05)
    Noise_object['x2'] = max([HC_object['x1'],HC_object['x2']])
    Noise_object['y1'] = min([LP_object['y1'],LP_object['y2']])
    Noise_object['y2'] = max([LP_object['y1'],LP_object['y2']])
    
    objects.append(Noise_object)
    
    offset = int(np.mean([LP_object['y1'],LP_object['y2']]) - np.mean([HC_object['y1'],HC_object['y2']]))
    
    LC_object=({'type':'4','position':'south'})
    LC_object['x1'] = HC_object['x1']
    LC_object['x2'] = HC_object['x2']
    LC_object['y1'] = HC_object['y1']+2*offset
    LC_object['y2'] = HC_object['y2']+2*offset
    
    objects.append(LC_object)
    
    # invert the image if needed, now that we have the objects
    mean_LP = np.mean(image[LP_object['y1']:LP_object['y2'],LP_object['x1']:LP_object['x2']])
    mean_noise = np.mean(image[Noise_object['y1']:Noise_object['y2'],Noise_object['x1']:Noise_object['x2']])
    if mean_LP > mean_noise:
        transformations.append('Inverting the image')
        image = np.amax(image)-image
    else:
        transformations.append('Inverting not needed')
    
    colors={'1':'red','2':'yellow','3':'blue','4':'green'}

    print('  4. build figure')
    fig = plt.figure()
    plt.imshow(image,cmap='gray')
    ax = plt.gca()
    for object in objects:
        x1 = int(object['x1'])
        x2 = int(object['x2'])
        y1 = int(object['y1'])
        y2 = int(object['y2'])
        width, height = x2 - x1, y2 - y1
        rect = Rectangle((x1, y1), width, height, fill=False, color=colors[object['type']])
        ax.add_patch(rect)
    plt.title('Image with objects')

    fn = 'image_with_objects.png'
    fig.savefig(fn, bbox_inches='tight')
    results.addObject('image_with_objects', fn)
    plt.show()
    plt.close(fig)

    results.addString('image_transformations', str(transformations))
    
    return image,objects