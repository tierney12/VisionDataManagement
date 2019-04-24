"""
This module contains miscellaneous controller functions that have been
abstracted away from the main controller file. This module mostly contains I/O operations for
loading and storing the RetinaImageVector objects in a variety of file types.

Author: Sean P. Tierney
Date: April 2019

"""

import sys
import os
from functools import partial
from os.path import join
import re
import time
import multiprocessing as mp
import numpy as np
import pandas as pd
import cv2
import h5py
import GPUtil
import sys
from pprint import pprint

#from PyQt5 import QtGui, QtCore

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QImage, QPixmap

from model import ImageVector
from retinavision.retina import Retina
from retinavision.cortex import Cortex
from retinavision import datadir, utils


def isGPUAvailable():
    DEVICE_LIST = [gpu.name for gpu in GPUtil.getGPUs()]
    print(DEVICE_LIST)
    return ["GeForce" in d for d in DEVICE_LIST]

def prepareLiveRetina(cap, yval, xval):
    """Performs extra preparation steps for using the retina with a webcam

    Parameters
    ----------
    cap : VideoCapture object capturing the video feed
    """

    print(yval)
    print(xval)
    retina = startRetina()
    ret, frame = cap.read()
    # fixation points inputted from form
    x = frame.shape[1]/((yval/360))
    y = frame.shape[0]/((xval/480))
    fixation = (y, x)
    #pprint(frame.shape)
    retina.prepare(frame.shape, fixation)
    return retina, fixation

def startRetina():
    """Instantiates Retina object and loads necessary files"""
    retina = Retina(isGPUAvailable())
    retina.loadLoc(join(datadir, "retinas", "ret50k_loc.pkl"))
    retina.loadCoeff(join(datadir, "retinas", "ret50k_coeff.pkl"))
    return retina

def getBackProjection(R, V, shape, fix):
    """Proxy function for Retina.backproject()"""
    return R.backproject(V,shape,fix)

def getBackProjections(frame, R):
    """Gets backprojection given only an image. Similar to a Python partial function"""
    #R = startRetina()
    backshape = (720,1280, frame.vector.shape[-1])
    return getBackProjection(R, frame._vector, backshape, fix=(frame.fixationy, frame.fixationx))

def createCortex():
    """Instantiates Cortex object and loads necessary files"""
    cortex = Cortex(isGPUAvailable())
    lp = join(datadir, "cortices", "50k_Lloc_tight.pkl")
    rp = join(datadir, "cortices", "50k_Rloc_tight.pkl")
    cortex.loadLocs(lp, rp)
    cortex.loadCoeffs(join(datadir, "cortices", "50k_Lcoeff_tight.pkl"), join(datadir, "cortices", "50k_Rcoeff_tight.pkl"))
    return cortex

def convertToPixmap(frame, x, y, BGR=False):
    """Converts images into QPixmap ojbects so they can be understood by UI elements

    Parameters
    ----------
    frame : image to be displayed
    x : width needed
    y : height needed
    BGR : boolean if colour space is BGR
    """

    if frame.shape[-1] == 3:
        if BGR:
            frame = cv2.cvtColor(frame,cv2.cv2.COLOR_BGR2RGB)
        format = QImage.Format_RGB888
    # image is grayscale (1 channel)
    else:
        format = QImage.Format_Grayscale8
    # convert first to QImage...

    print(frame.data)
    converttoQtFormat = QImage(frame.data,frame.shape[1],frame.shape[0],format)
    # ...scale result...
    pic = converttoQtFormat.scaled(x,y,Qt.KeepAspectRatio)
    # ...and then convert to QPixmap
    pixmap = QPixmap.fromImage(pic)
    return pixmap

def createImagesFromFolder(currentDir):
    """Scans a given directory and generates Image objects for all images found

    Parameters
    ----------
    currentDir : selected directory for search
    """

    currentFrames = []
    count = 1
    for root, dirs, files in os.walk(currentDir):
        for file in files:
            print(file)
            filetype = file.split(".")[-1]
            if filetype in {'jpg','png'}:
                print("Creating image object")
                image = cv2.imread(join(root,file))
                frame = Image(image=image,filepath=join(root,file),framenum=count)
                currentFrames.append(frame)
                count += 1
    return currentFrames

def createImageVectorsFromFolder(currentDir):
    """Scans a given directory and generates Image objects for all images found

    Parameters
    ----------
    currentDir : selected directory for search
    """

    currentFrames = []
    count = 1
    for root, dirs, files in os.walk(currentDir):
        for file in files:
            print(file)
            filetype = file.split(".")[-1]
            if filetype in {'jpg','png'}:
                print("Creating imagevector object")
                image = cv2.imread(join(root,file))
                frame = ImageVector(image, name='image' + str(count), vectortype='raw', framenum=count, id=file)
                currentFrames.append(frame)
                count += 1
    return currentFrames

def loadhdf5(filename, frames):
    """Loads a HDF5 data file and generates ImageVector objects for each record

    Parameters
    ----------
    filename : specified source name
    frames : existing frames in system (should probably pass something lighter)
    """

    currentframes = frames if frames else []
    hdf5_open = h5py.File(filename, mode="r")


    if 'vector' in hdf5_open.keys():
        for i in range(len(hdf5_open['vector'])):
            try:
                # if metadata is available, try and load it
                if 'retinatype' in hdf5_open.keys():
                    v = ImageVector(
                        vector=hdf5_open['vector'][i],
                        label=hdf5_open['label'][i].tostring(),
                        fixationy=int(hdf5_open['fixationy'][i]),
                        fixationx=int(hdf5_open['fixationx'][i]),
                        retinatype=hdf5_open['retinatype'][i].tostring(),
                        vectortype=hdf5_open['vectortype'][i].tostring(),
                        id=hdf5_open['id'][i].tostring(),
                        name=hdf5_open['name'][i].tostring())
                # if not, load just the imagevectors
                else:
                    v = ImageVector(vector=hdf5_open['vector'][i])
                # load these metadata items if available, else an alternative
                v.framenum = int(hdf5_open['framenum'][i]) if 'framenum' in hdf5_open.keys() else None
                v._timestamp = hdf5_open['timestamp'][i].tostring() if 'timestamp' in hdf5_open.keys() else None
                v.vectortype = hdf5_open['vectortype'][i].tostring() if 'vectortype' in hdf5_open.keys() else "unknown"
            except:
                raise Exception('HDF5Format')
            currentframes.append(v)
        print("Vector shape: " + str(currentframes[0]._vector.shape))
        # use multiprocessing for generating backprojections
    else:
        raise Exception('HDF5Format')

    return currentframes

def saveHDF5(exportname, frames):
    """Creates a h5py File object and stores the current frames within it

    Parameters
    ----------
    exportname : specified name (and location) of output file
    frames : list of frames to be stored
    """

    vectors, labels, framenums, timestamps, fixationY, fixationX, retinatypes, vectortypes, ids, names = ([] for i in range(10))
    hdf5_file = h5py.File(exportname, mode='w')
    currentframe = None
    # extract attributes into separate lists
    for frame in frames:
        vectors.append(frame._vector)
        labels.append(frame.label)
        framenums.append(0 if frame.framenum is None else int(frame.framenum))
        timestamps.append("unknown" if frame._timestamp is None else str(frame._timestamp))
        fixationY.append(0 if frame.fixationy is None else int(frame.fixationy))
        fixationX.append(0 if frame.fixationx is None else int(frame.fixationx))
        retinatypes.append("unknown" if frame.retinatype is None else str(frame.retinatype))
        vectortypes.append("unknown" if frame.vectortype is None else str(frame.vectortype))
        ids.append("unknown" if frame.id is None else str(frame.id))
        names.append("unknown" if frame.name is None else str(frame.name))


    # create datasets in new file with appropriate data types
    hdf5_file.create_dataset("vector",(len(vectors),len(frame._vector),frame._vector.shape[-1]),np.float64)
    hdf5_file.create_dataset("label",(len(labels),1),"S200")
    hdf5_file.create_dataset("framenum",(len(labels),1),np.int16)
    hdf5_file.create_dataset("timestamp",(len(labels),1),"S11")
    hdf5_file.create_dataset("fixationy",(len(labels),1),np.int16)
    hdf5_file.create_dataset("fixationx",(len(labels),1),np.int16)
    hdf5_file.create_dataset("retinatype",(len(labels),1),"S100")
    hdf5_file.create_dataset("vectortype",(len(labels),1),"S100")
    hdf5_file.create_dataset("id", (len(labels), 1), "S100")
    hdf5_file.create_dataset("name", (len(labels), 1), "S100")

    # store data in new datasets
    for i in range(len(vectors)):
        hdf5_file["vector"][i] = vectors[i]
        hdf5_file["label"][i] = labels[i]
        hdf5_file["framenum"][i] = framenums[i]
        hdf5_file["timestamp"][i] = timestamps[i]
        hdf5_file["fixationy"][i] = fixationY[i]
        hdf5_file["fixationx"][i] = fixationX[i]
        hdf5_file["retinatype"][i] = retinatypes[i]
        hdf5_file["vectortype"][i] = vectortypes[i]
        hdf5_file["id"][i] = ids[i]
        hdf5_file["name"][i] = names[i]

    hdf5_file.close()

def loadCsv(filename, frames):
    """Loads a CSV data file and  either generates ImageVector objects for each
    record, or adds metadata to existing frames

    Parameters
    ----------
    filename : specified source name
    frames : existing frames in system (should probably pass something lighter)
    concurrbackproject : boolean for using multiprocessing to generate backprojections
    """

    metadata = pd.read_csv(filename,delimiter=";",encoding="utf-8")#,index_col="framenum"
    currentframes = frames if frames else []
    cols = metadata.columns
    print("Columns found: " + str(cols))
    count = 1
    for i in range(metadata.shape[0]):
        # interpreting the imagevector column
        if metadata['vector'][i].startswith('['):
            # if it's a multi-channel Imagevector, break around '[]'
            vtemp = re.findall("\[(.*?)\]", metadata['vector'][i])
            # break resulting strings again into string numbers
            for j in range(len(vtemp)):
                vtemp[j] = [x for x in vtemp[j].split(' ') if x]
            # convert string number lists into floating-point arrays
                vtemp = np.asarray([x for x in vtemp], dtype=np.float64)
                # convert list of lists into an array
                vector = np.asarray(vtemp,dtype=np.float64)
        else:
            # if it's a single-channel vector, just break around commas
            vector = np.asarray(metadata['vector'][i].split(","),
                dtype=np.float64)
        try:
            v = ImageVector(vector=vector,
                label=metadata['label'][i],
                fixationy=int(metadata['fixationy'][i]),
                fixationx=int(metadata['fixationx'][i]),
                retinatype=metadata['retinatype'][i])
            v.framenum = int(metadata['framenum'][i]) if 'framenum' in cols else count
            v._timestamp = metadata['timestamp'][i] if 'timestamp' in cols else None
            v.vectortype = metadata['vectortype'][i] if 'vectortype' in cols else None
        except:
            raise Exception('CSVFormat')
        count += 1
        currentframes.append(v)
        # Show user resulting array's shape for diagnostics
    return currentframes

def saveCSV(exportname, frames):
    """Creates a dataframe and stores the current frames within it, before
    converting to a CSV file

    Parameters
    ----------
    exportname : specified name (and location) of output file
    frames : list of frames to be stored
    """

    # get attribute names
    columns = dir(frames[0])
    # create a dataframe with columns named after attributes and store values
    df = pd.DataFrame([{fn: getattr(f,fn) for fn in columns} for f in frames])
    pprint(df)
    vectorstrings = []
    for frame in frames:
        print("Creating stringvector")
        # convert the imagevector into a string suitable for storage in CSV
        vs = ','.join(str(e) for e in frame._vector)
        vectorstrings.append(vs)
    df['vector'] = pd.Series(vectorstrings, index=df.index)
    df.rename(columns = {'_timestamp':'timestamp'}, inplace = True)
    # exported file should be read with ';' delimiter ONLY
    df.to_csv(exportname,encoding='utf-8',sep=";") # compression='gzip'?

def loadPickle(filename):
    return utils.loadPickle(filename)
