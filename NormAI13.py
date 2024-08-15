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
#   20240131: subpixel detection of LC objects
#
# python NormAI13.py -r results.json -c config.json -d images\studies\study_01

from __future__ import print_function

__version__ = '20240206'
__author__ = 'jmgroen'

import os
from wad_qc.module import pyWADinput
from wad_qc.modulelibs import wadwrapper_lib
import numpy as np
from datetime import datetime


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
        try:
           results.addFloat(varname, float(value))
        except:
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

    try:
        dt = wadwrapper_lib.get_datetime(dcmInfile,"Acquisition")
        results.addDateTime('AcquisitionDateTime', dt)
    except:
        print('dicom file does not contain the acquistion datetime, setting it to now.')
        current_datetime = datetime.now()
#        dt = current_datetime.strftime('%Y-%m-%d %H:%M:%S')
        results.addDateTime('AcquisitionDateTime', current_datetime)  

from NormAI13_processNoise import process_Noise
from NormAI13_processLP import process_LP
from NormAI13_processLC import process_LowContrast
from NormAI13_processHC import process_HC
from NormAI13_detection import normi13_detection
from NormAI13_processdetection import process_detection

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
    
    instance=instances[0]

    print('Run detection')
    start = time.time()
    image,detection,resolution = normi13_detection(instance,results)
    print('Detection completed in',time.time()-start,'seconds')
    print('')

    print('Process detection')
    start = time.time()
    image,objects = process_detection(image,detection,results)
    print('Processing completed in',time.time()-start,'seconds')
    print('')
    
    for object in objects:
        object_image = image[object['y1']:object['y2'],object['x1']:object['x2']]
        if object['type'] == '1':
            print('Analyzing HC')
            try:
                start = time.time()
                completed = process_HC(object_image,results)
                print('HC analysis completed in',time.time()-start,'seconds')
                print('')
            except Exception as e:
                print('HC analysis failed:',e)
                print('')
        if object['type'] == '2':
            print('Analyzing LP')
            try:
                start = time.time()
                completed = process_LP(object_image,resolution,results)
                print('LP analysis completed in',time.time()-start,'seconds')
                print('')
            except Exception as e:
                print('LP analysis failed:',e)
                print('')
        if object['type'] == '3':
            print('Analyzing Noise')
            try:
                start = time.time()
                completed = process_Noise(object_image,results)
                print('Noise analysis completed in',time.time()-start,'seconds')
                print('')
            except Exception as e:
                print('Noise analysis failed:',e)
                print('')
        if object['type'] == '4':
            print('Analyzing LC')
            try:
                start = time.time()
                result = process_LowContrast(object_image,resolution,results)
                print('LC analysis completed in',time.time()-start,'seconds')
                print('')
            except Exception as e:
                print('LC analysis failed:',e)
                print('')
                

    
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
