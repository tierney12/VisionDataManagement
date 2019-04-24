"""
This file contains the ImageVector (also referred to as RetinaImageVector) class that is the
data structure used to store imagevectors and their associated metadata.

This class also contains the Video class that is the model used to store the data used
by the VideoPlayer class in customwidgets.py

Author: Sean P. Tierney
Date: April 2019
"""

import sys
import os
import csv
import cv2
import time
import pprint
from os.path import join
from pprint import pprint

class ImageVector(object):
    """Represents an imagevector and its associated metadata"""
    def __init__(self, vector=None, id=None, framenum=None, timestamp=time.time(), label=None, fixationy=None, fixationx=None, retinatype=None, vectortype=None, name=None):
        self._vector = vector #the imagevector, size can be easily obtained with func size(_vector)
        self.id = id

        self.framenum = framenum
        self._timestamp = timestamp
        self.label = label #this is a single string, may be a comma separated list
        self.fixationx = fixationx
        self.fixationy = fixationy
        self.retinatype = retinatype
        self.vectortype = vectortype #stores if the image is raw/backprojected/cortical
        self.name = name

    def __dir__(self):
        return ['id','framenum','_timestamp','label','fixationy','fixationx','retinatype','vectortype','name']

'''
#deprecated, now use Imagevector in all cases
class Image(object):
    """Represents an image file and its associated metadata """
    def __init__(self, image=None, name=None, filepath=None, colortype=None, parent=None, framenum=None, label=None):

        #self.vector = None
        #self.vectortype = None
        #self.type = None

        self.image = image
        self.name = name
        if filepath:
            self.type = filepath.split('.')[-1]
            if not name:
                self.name = filepath.split("/")[-1]
        self.filepath = filepath
        self.parent = parent
        self.framenum = framenum
        self.colortype = colortype
        self.label = label

    def saveImageOnly(self,dir):
        """Save the image back to file if required"""
        print("Saving image")
        cv2.imwrite(join(dir,"frame%d.png" % self.framenum), self.image)

    def __dir__(self):
        return ['name','type','framenum','colortype','label']
'''

class Video(object):
    """Represents a video file, its metadata and individual frames if generated"""
    filetypes = {
        'mp4': 'mp4v',
        'avi': 'xvid',
        'MP4': 'MP4V'
    }

    def __init__(self,filepath,colortype):
        self.name = filepath.split("/")[-1]
        self.filepath = filepath
        self.type = filepath.split('.')[-1]
        self.colortype = colortype
        self.frames = None # list of Images, filled upon user request
        self.numFrames = None

    def getFrames(self):
        """Use OpenCV to split the video file into frames and capture them as Image objects"""
        self.cap = cv2.VideoCapture(self.filepath)
        codec = cv2.VideoWriter_fourcc(*self.filetypes[self.type])
        self.cap.set(cv2.CAP_PROP_FOURCC, codec)
        self.numFrames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.frames = [0] * self.numFrames
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                framenum = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
                self.frames[framenum - 1] = Image(frame,parent=self,framenum=framenum,name="frame"+str(framenum))
            else:
                break
        print("Images are now in memory at self.frames. Save them to disk by calling a version of saveFrames")
        return self.frames

    def getFramesImageVectors(self):
        """Use OpenCV to split the video file into frames and capture their ImageVector objects"""
        print('enters here')
        self.cap = cv2.VideoCapture(self.filepath)
        count = 0
        #codec = cv2.VideoWriter_fourcc(*self.filetypes[self.type])
        #self.cap.set(cv2.CAP_PROP_FOURCC, codec)
        #self.numFrames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        #self.frames = [0] * self.numFrames
        self.frames = []

        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                #self.frames[framenum - 1] = Image(frame,parent=self,framenum=framenum,name="frame"+str(framenum))
                self.frames.append(ImageVector(vector=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), name='frame' + str(count), framenum=count))
                count += 1
            else:
                break
        print("Images are now in memory at self.frames. Save them to disk by calling a version of saveFrames")

        return self.frames

    def saveFramesImageOnly(self):
        """Saves the images back to file if required"""
        frames_dir = join(self.filepath.split("/")[0], "Frames")
        if not os.path.exists(frames_dir):
            os.makedirs(frames_dir)
        if self.frames:
            for frame in self.frames:
                if frame:
                    frame.saveImageOnly(frames_dir)
        else:
            print("Frames not yet captured from video. Try self.getFrames()")
