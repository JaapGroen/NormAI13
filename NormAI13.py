from wad_qc.module import pyWADinput
from wad_qc.modulelibs import wadwrapper_lib
import numpy as np


def logTag():
    return "[NormAI13] "
    
def LogHeaderTags(data, results, action):
       
    # get the first (and only) file
    instances = data.getAllInstances()
    
    instance=instances[0]
        
    # we need pydicom to read out dicom tags
    try:
        import pydicom as dicom
    except ImportError:
        import dicom
           
    # look in the config file for tags and write them as results, nested tags are supported 2 levels
    for key in action['tags']:
        varname=key
        tag=action['tags'][key]
        try:
            if tag.count('/')==0:
                value=instance[dicom.tag.Tag(tag.split(',')[0],tag.split(',')[1])].value
            elif tag.count('/')==1:
                tag1=tag.split('/')[0]
                tag2=tag.split('/')[1]
                value=instance[dicom.tag.Tag(tag1.split(',')[0],tag1.split(',')[1])][0]\
                [dicom.tag.Tag(tag2.split(',')[0],tag2.split(',')[1])].value
            elif tag.count('/')==2:
                tag1=tag.split('/')[0]
                tag2=tag.split('/')[1]
                tag3=tag.split('/')[2]
                value=instance[dicom.tag.Tag(tag1.split(',')[0],tag1.split(',')[1])][0]\
                [dicom.tag.Tag(tag2.split(',')[0],tag2.split(',')[1])][0]\
                [dicom.tag.Tag(tag3.split(',')[0],tag3.split(',')[1])].value
            else:
                # not more then 2 levels...
                value='too many levels'

        except:
            value = 'Tag missing from header'
            
        # write results
        results.addString(varname, str(value)[:min(len(str(value)),100)])



def acqdatetime_series(data, results, action):
    """
    Read acqdatetime from dicomheaders and write to IQC database

    Workflow:
        1. Read only headers
    """
    try:
        import pydicom as dicom
    except ImportError:
        import dicom
    try:
        params = action['params']
    except KeyError:
        params = {}    
        
    ## 1. read only headers
    dcmInfile = dicom.read_file(data.series_filelist[0][0], stop_before_pixels=True)
    
    ds = dicom.dcmread(data.series_filelist[0][0])
    print('Syntax:',ds.file_meta.TransferSyntaxUID)
    
#     print(dcmInfile)

    dt = wadwrapper_lib.get_datetime(dcmInfile,"Acquisition")

    results.addDateTime('AcquisitionDateTime', dt)
    
#     print('StudyDate:',dt)        

def Normi13_analysis(data, results, action):

    import time
    
    try:
        params = action['params']
    except KeyError:
        params = {}
    
    # assume that there is 1 file with multiple images
    instances = data.getAllInstances()
    
    if len(instances)>1:
        print('There are more than 1 images, only analyzing the first one!')
        # print('File:',instances[0].filename)
    
    
    
#     imgs = []
#     for instance in instances:
#         new_img = instance.pixel_array.astype('float32')
# #         hash = imagehash.average_hash(new_img)
#         print(hash(str(new_img)))
        

#     print(len(imgs),' unique images found')
            
    
    
    instance=instances[0]
       
    main_results = {}
    def process_results(result,prefix=None):
        for item in result:
            main_results[item] = result[item]
            if prefix:
                print(prefix,':',item,'-',result[item])
        return True
    
    # objection detection on the raw dicom input
    from NormAI13_detection import normi13_detection
    print('Start detection')
    start = time.time()
    result = normi13_detection(instance)
    print('Detection completed in',time.time()-start,'seconds')
    process_results(result)
    fn = 'Detected_objects.png'
    main_results['detection_fig'].savefig(fn, bbox_inches='tight')
    results.addObject('Detected_objects', fn) 
    results.addString('Pixelsize', str(main_results['resolution'][0])+' x '+str(main_results['resolution'][1]))
    
    # image transformations and determination of other objects
    from NormAI13_processdetection import process_detection
    print('Start processing')
    start = time.time()
    result = process_detection(main_results['image'],main_results['detection'])
    print('Processing completed in',time.time()-start,'seconds')
    process_results(result)
    fn = 'Processing_ROIs.png'
    main_results['image_objects'].savefig(fn, bbox_inches='tight')
    results.addObject('Processing_ROIs', fn)
    results.addString('Image_transformations', str(main_results['transformations']))
    
    # analyze the different objects
    from NormAI13_processNoise import process_Noise
    from NormAI13_processLP import process_LP
    from NormAI13_processLC import process_LowContrast
    from NormAI13_processHC import process_HC
    
    for object in main_results['objects']:
        object_image = main_results['image'][object['y1']:object['y2'],object['x1']:object['x2']]
        if object['type'] == '1':
            print('Start HC analysis')
            try:
                start = time.time()
                result = process_HC(object_image)
                print('HC analysis completed in',time.time()-start,'seconds')
                if process_results(result):
                    results.addFloat('HighContrast visible elements', main_results['num_elem'])
                    fn = 'HighContrast_plot.png'
                    main_results['HighContrast_plot'].savefig(fn, bbox_inches = 'tight')
                    results.addObject('HighContrast_plot', fn)
                    fn = 'HighContrast_wedge_image.png'
                    main_results['corrected_image'].savefig(fn, bbox_inches = 'tight')
                    results.addObject('HighContrast_wedge_image', fn)
            except Exception as e:
                print('HC analysis failed:',e)
        if object['type'] == '2':
            print('Start LP analysis')
            try:
                start = time.time()
                result = process_LP(object_image,main_results['resolution'])
                print('LP analysis completed in',time.time()-start,'seconds')
                if process_results(result):                  
                    results.addFloat('Phantom angle', main_results['correction angle']-135)
                    results.addFloat('Resolution visible', main_results['resolution visible'])
                    fn = 'LinePairs_corrected_image.png'
                    main_results['corrected image'].savefig(fn, bbox_inches = 'tight')
                    results.addObject('LinePairs_corrected_image', fn)
                    fn = 'LinePairs_cropped_bars.png'
                    main_results['image_bars'].savefig(fn, bbox_inches = 'tight')
                    results.addObject('LinePairs_cropped_bars', fn)
                    fn = 'LinePairs_profiles.png'
                    main_results['bar_profiles'].savefig(fn, bbox_inches = 'tight')
                    results.addObject('LinePairs_profiles', fn)
            except Exception as e:
                print('LP analysis failed:',e)
        if object['type'] == '3':
            print('Running Noise analysis')
            try:
                start = time.time()
                result = process_Noise(object_image)
                print('Noise analysis completed in',time.time()-start,'seconds')
                if process_results(result):
                    results.addFloat('Maximum deviation', main_results['maximum deviation'])
                    results.addFloat('Uniformity', main_results['uniformity'])
                    results.addFloat('SNR', main_results['snr'])
                    fn = 'Uniformity_ROIs.png'
                    main_results['Uniformity image'].savefig(fn, bbox_inches = 'tight')
                    results.addObject('Uniformity_ROIs', fn)
                    results.addFloat('Uniformity_A', main_results['uniformity_A'])
                    results.addFloat('Uniformity_B', main_results['uniformity_B'])
            except Exception as e:
                print('Noise analysis failed:',e)
        if object['type'] == '4':
            print('Running LC analysis')
            try:
                start = time.time()
                result = process_LowContrast(object_image,main_results['resolution'])
                print('LC analysis completed in',time.time()-start,'seconds')
                if process_results(result):
                    fn = 'LowContrast_ROIs.png'
                    main_results['image with ROIs'].savefig(fn, bbox_inches = 'tight')
                    results.addObject('LowContrast_ROIs', fn)
                    fn = 'LowContrast_plot.png'
                    main_results['LowContrast_plot'].savefig(fn, bbox_inches = 'tight')
                    results.addObject('LowContrast_plot', fn)

                    fn = 'Object1_detection.png'
                    main_results['Object1_detection'].savefig(fn, bbox_inches = 'tight')
                    results.addObject('Object1_detection', fn)
                    fn = 'Object2_detection.png'
                    main_results['Object2_detection'].savefig(fn, bbox_inches = 'tight')
                    results.addObject('Object2_detection', fn)
            except Exception as e:
                print('LC analysis failed:',e)
                

    
if __name__ == "__main__":
    #import the pyWAD framework and get some objects
    data, results, config = pyWADinput()

    for name,action in config['actions'].items():
        if name == 'acqdatetime':
            acqdatetime_series(data, results, action)

        elif name == 'LogHeaderTags':
            LogHeaderTags(data, results, action)

        elif name == 'Normi13_analysis':
            Normi13_analysis(data, results, action)
            
        results.write()