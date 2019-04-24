"""
This module contains subclasses of PyQt5 widgets used to implement the highlighting feature of the image gallery
as well as the VideoPlayer class.

Author: Sean P. Tierney
Date: April 2019
"""

import PyQt5
from PyQt5.QtWidgets import *
from PyQt5.QtCore import pyqtSignal, QTimer
from model import Video
import time
import processing as ip
import cv2
import threading
from pprint import pprint


class ClickLabel(QLabel):
    """Adds on-click functionality to Qt's QLabel"""
    clicked=pyqtSignal(int)
    unhighlighted=pyqtSignal(int)

    def __init__(self, parent=None, index=0):
        QLabel.__init__(self, parent)
        self.index = index
        self.highlighted = False

    def mousePressEvent(self, event):
        self.highlighted = not self.highlighted
        if self.highlighted:
            self.makeHighlighted()
            self.clicked.emit(self.index)
        else:
            self.notHighlighted()
            self.unhighlighted.emit(self.index)

    def setIndex(self, index):
        self.index = index

    def makeHighlighted(self):
        self.setFrameShape(PyQt5.QtWidgets.QFrame.Box)
        self.setLineWidth(3)

    def notHighlighted(self):
        self.setFrameShape(PyQt5.QtWidgets.QFrame.StyledPanel)
        self.setLineWidth(1)


class VideoPlayer(QWidget):
    """Enables playback functions and broadcasting of video stream to UI elements

    Subscribers are named individually because they require different images or data
    """

    video = None
    videoFrame = None
    focalFrame = None
    corticalFrame = None
    focusFrame = None
    framePos = 0
    maxFrames = 0
    retina = None
    cortex = None
    fixation = None
    filetypes = {
        'mp4': 'mp4v',
        'jpg': 'jpeg',
        'avi': 'xvid',
        'MP4': 'x264'
    }

    def __init__(self,file,isRetinaEnabled,parent,webcammode=False):
        super(QWidget,self).__init__()
        self.parent = parent
        self.isVideo = False
        self.webcam = webcammode
        self.isRetinaEnabled = isRetinaEnabled
        if file:
            self.file = file
            self.isVideo = isinstance(file,Video)
        self.timer = QTimer()
        self.timer.timeout.connect(self.nextFrame)
        self.frames = parent.currentFrames
        # set up video capture dependent on source
        if self.isVideo:
            self.cap = cv2.VideoCapture(self.file.filepath)
            pprint(self.file.type)
            codec = cv2.VideoWriter_fourcc(*self.filetypes[self.file.type])
            self.cap.set(cv2.CAP_PROP_FOURCC, codec)
            self.maxFrames = self.cap.get(cv2.CAP_PROP_FRAME_COUNT)
        # if webcam mode, start immediately
        elif self.webcam:
            self.cap = cv2.VideoCapture(0)
            self.isBGR = True
            self.timer.start(1000.0/30) #framerate
        # if still images, no need for video capture
        else:
            self.maxFrames = len(self.frames) - 1
        self.parent.scrubSlider_2.setRange(0,self.maxFrames)
        self.framePos = 0
        self.videoFrame = parent.label
        self.focalFrame = parent.focalLabel
        self.corticalFrame = parent.corticalLabel
        self.focusFrame = parent.biglabel
        if isRetinaEnabled:
            self.retina, self.fixation = ip.prepareLiveRetina(self.cap, self.parent.yspinBox.value(), self.parent.xspinBox.value())
            self.cortex = ip.createCortex()


    def nextFrame(self):
        """Retrieves next frame for display whether video or image"""
        print(self.maxFrames)
        if self.isVideo or self.webcam:
            ret, frame = self.cap.read()
            if ret:
                self.framePos = self.cap.get(cv2.CAP_PROP_POS_FRAMES)
                self.updateDisplay(frame)
        else:
            self.framePos += 1 if self.framePos < self.maxFrames else 0
            self.setCurrent()

    def start(self):
        self.timer.start(self.parent.delaySpinBox.value()*1000)
        self.parent.startButton.setDisabled(True)
        self.parent.startButton_2.setDisabled(True)
        self.parent.pauseButton.setDisabled(False)
        self.parent.pauseButton_2.setDisabled(False)

    def pause(self):
        self.timer.stop()
        self.parent.pauseButton.setDisabled(True)
        self.parent.pauseButton_2.setDisabled(True)
        self.parent.startButton.setDisabled(False)
        self.parent.startButton_2.setDisabled(False)

    def setCurrent(self):
        """Sets the current frame based on user input from playback buttons"""
        self.maxFrames = len(self.frames)
        print(str(self.framePos) + ' of ' + str(self.maxFrames))
        if self.framePos < self.maxFrames:
            if self.isVideo:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.framePos)
            else:
                self.currentframe = self.frames[self.framePos]
                #pprint(self.currentframe)
                self.updateDisplay(self.currentframe._vector)

    def skip(self, framePos):
        print('skip')
        self.framePos = framePos
        self.setCurrent()

    def skipBck(self):
        self.framePos = self.framePos - 1 if self.framePos > 0 else 0
        self.setCurrent()

    def skipFwd(self):
        self.framePos = self.framePos + 1 if self.framePos < self.maxFrames else 0
        self.setCurrent()

    def skipToFirst(self):
        print(self.framePos)
        self.framePos = 0
        self.setCurrent()

    def skipToLast(self):
        self.framePos = self.maxFrames
        self.setCurrent()

    def updateDisplay(self, frame):
        """Update all subscribed UI elements with images or data"""
        self.parent.scrubSlider.setValue(self.framePos)
        self.parent.scrubSlider_2.setValue(self.framePos)
        self.parent.frameNum.display(self.framePos)
        self.parent.frameNum_2.display(self.framePos)
        # update the metadata table if we're on the main tab
        if self.parent.maintabWidget.currentIndex() == 1:
            self.parent.displayMetaData(self.framePos)
        if not (self.isVideo or self.webcam):
            self.isBGR = self.currentframe.vectortype == 'BGR'
        self.videoFrame.setPixmap(ip.convertToPixmap(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), 480, 360))
        # if retina is activated display live backprojection and cortical image
        if self.isRetinaEnabled:
            v = self.retina.sample(frame, (self.ysspinBox.value(), self.xspinBox.value()))
            tight = self.retina.backproject_last()
            cortical = self.cortex.cort_img(v)
            self.focalFrame.setPixmap(ip.convertToPixmap(cv2.cvtColor(tight, cv2.COLOR_BGR2RGB), 480, 360))
            self.corticalFrame.setPixmap(ip.convertToPixmap(cortical, 480, 360, self.isBGR))
            self.focusFrame.setPixmap(ip.convertToPixmap(cortical, 1280, 720, self.isBGR))
        else:
            self.focusFrame.setPixmap(ip.convertToPixmap(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), 1280, 720))

class VideoCaptureAsync:
    def __init__(self, src=0, width=640, height=480):
        self.src = src
        self.cap = cv2.VideoCapture(self.src)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.grabbed, self.frame = self.cap.read()
        self.started = False
        self.read_lock = threading.Lock()

    def set(self, var1, var2):
        self.cap.set(var1, var2)

    def start(self):
        if self.started:
            print('[!] Asynchroneous video capturing has already been started.')
            return None
        self.started = True
        self.thread = threading.Thread(target=self.update, args=())
        self.thread.start()
        return self

    def update(self):
        while self.started:
            grabbed, frame = self.cap.read()
            with self.read_lock:
                self.grabbed = grabbed
                self.frame = frame

    def read(self):
        with self.read_lock:
            frame = self.frame.copy()
            grabbed = self.grabbed
        return grabbed, frame

    def stop(self):
        self.started = False
        self.thread.join()

    def __exit__(self, exec_type, exc_value, traceback):
        self.cap.release()