#!/usr/bin/env python
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
# PyWAD is open-source software and consists of a set of modules written in python for the WAD-Software medical physics quality control software. 
# The WAD Software can be found on https://github.com/wadqc
# 
# The pywad package includes modules for the automated analysis of QC images for various imaging modalities. 
# PyWAD has been originaly initiated by Dennis Dickerscheid (AZN), Arnold Schilham (UMCU), Rob van Rooij (UMCU) and Tim de Wit (AMC) 
#
#
# Changelog:
#   20200417: first version
#
# python NormAI13.py -r results.json -c config.json -d images\studies\study_01

from __future__ import print_function

__version__ = '20200417'
__author__ = 'jmgroen'

import os
from wad_qc.module import pyWADinput
from wad_qc.modulelibs import wadwrapper_lib

import numpy as np
import scipy
if not 'MPLCONFIGDIR' in os.environ:
    # using a fixed folder is preferable to a tempdir, because tempdirs are not automatically removed
    os.environ['MPLCONFIGDIR'] = "/tmp/.matplotlib" # if this folder already exists it must be accessible by the owner of WAD_Processor 
import matplotlib
matplotlib.use('Agg') # Force matplotlib to not use any Xwindows backend.

# imports
import torch
import torchvision

def logTag():
    return "[NormAI13] "

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

    dt = wadwrapper_lib.acqdatetime_series(dcmInfile)

    results.addDateTime('AcquisitionDateTime', dt)
     

def header_series(data, results, action):
       
    # get the first (and only) file
    instances = data.getAllInstances()
    
    if len(instances) != 1:
        print('%s Error! Number of instances not equal to 1 (%d). Exit.'%(logTag(),len(instances)))
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

        # write results
        results.addString(varname, str(value)[:min(len(str(value)),100)])    
    
              
def Normi13_analysis(data, results, action):

    import time
    
    
    try:
        params = action['params']
    except KeyError:
        params = {}
    
    # assume that there is 1 file with multiple images
    instances = data.getAllInstances()
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
            except:
                print('HC analysis failed')
        if object['type'] == '2':
            print('Start LP analysis')
            try:
                start = time.time()
                result = process_LP(object_image,main_results['resolution'])
                print('LP analysis completed in',time.time()-start,'seconds')
                if process_results(result):
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
            except:
                print('LP analysis failed')
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
            except:
                print('Noise analysis failed')
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
            except:
                print('LC analysis failed')
                

    
if __name__ == "__main__":
    #import the pyWAD framework and get some objects
    data, results, config = pyWADinput()

    # look in the config for actions and run them
    for name,action in config['actions'].items():
        
        # save acquisition time and date as result        
        if name == 'acqdatetime':
           acqdatetime_series(data, results, action)

        # save whatever tag is requested as result
        elif name == 'header_series':
           header_series(data, results, action)

        # run the Normi13 analysis
        elif name == 'qc_series':
            Normi13_analysis(data, results, action)

    results.write()

    # all done