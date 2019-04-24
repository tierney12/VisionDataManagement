"""
This module is the primary controller that contains the application logic as well as
tying the UI elements to their respective functions.

Author: Sean P. Tierney
Date: April 2019


"""

import sys
import cv2
from cv2 import *
import design
import GPUtil
import processing as ip
import PyQt5
import os
import numpy
from customwidgets import VideoPlayer, ClickLabel, VideoCaptureAsync
from model import *
from worker import Worker

from retinavision.retina import Retina
from retinavision.cortex import Cortex
from retinavision import datadir, utils
from retinavision import utils

from PyQt5.uic import loadUi
from os.path import join

from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
#extension
from pprint import pprint
import qimage2ndarray

"""
This is the class that represents the running application, variables declared with self. are instance variables
available throughout the system. 
"""
class DMApp(QMainWindow, design.Ui_MainWindow):

    fileName = None
    currentDir = None
    currentFile = None
    currentFrames = None
    metaFileName = None
    metadatamodel = None
    videoPlayer = None
    isRetinaEnabled = False
    highlightedframes = []
    videofiletypes = {'mp4','avi','MP4'}
    metadatatypes = {'csv'}
    rawvectortypes = {'npy','npz'}

    def __init__(self, parent=None):
        super(DMApp, self).__init__(parent)
        self.setupUi(self)       # on load, connect UI elements to their respective functions
        self.startButton.clicked[bool].connect(self.runWebCam)
        self.pauseButton.clicked.connect(self.stop_webcam)
        self.fixationMouseButton.clicked[bool].connect(self.setFixationPoint)
        self.browseButton.clicked.connect(self.openFileNameDialog)
        self.browseFolderButton.clicked.connect(self.openFolderDialog)
        self.retinaButton.clicked.connect(self.setRetinaEnabled)
        self.generateButton.clicked.connect(self.getVideoFrames)
        self.exportButton.clicked.connect(self.saveFileDialog)
        self.saveButton.clicked.connect(self.saveMetaData)
        self.deleteButton.clicked.connect(self.deleteFrame)
        self.actionExport.triggered.connect(self.saveFileDialog)
        self.actionFile.triggered.connect(self.openFileNameDialog)
        self.actionFolder.triggered.connect(self.openFolderDialog)
        self.actionClose.triggered.connect(self.closeFile)
        self.actionSelect_All.triggered.connect(self.selectAll)
        self.actionDelete_Selection.triggered.connect(self.deleteFrame)
        self.actionExit.triggered.connect(self.closeApp)
        self.labels = self.dataframe_2.findChildren(ClickLabel)

        self.xspinBox.valueChanged.connect(self.prepare_retina)
        self.yspinBox.valueChanged.connect(self.prepare_retina)

        self.cap = VideoCaptureAsync()

        # sort gallery items by name so they can be filled easily
        self.labels.sort(key=lambda label: label.objectName())
        # connect gallery item click signals to their respective functions
        for i in range(len(self.labels)):
            self.labels[i].clicked.connect(self.displayMetaData)
            self.labels[i].unhighlighted.connect(self.removeHighlighted)
        self.numbers = self.dataframe_2.findChildren(PyQt5.QtWidgets.QLCDNumber)
        # also sort the frame number labels
        self.numbers.sort(key=lambda number: number.objectName())
        self.maintabWidget.setCurrentIndex(0)
        # instantiate a thread pool
        self.threadpool = QThreadPool()

        self.labelRecording.hide()
        self.pauseButton.setEnabled(False)
        self.recordButton.setEnabled(False)
        self.stopRecordButton.setEnabled(False)

        self.flagRecord = False
        
        self.image = None

        self.frames = []
        self.currentFrames = []
        self.height = 720
        self.width = 1080
        self.framerate = 30
        self.timer = None

        self.setup = False

        self.pauseButton.clicked.connect(self.stop_webcam)

        self.recordButton.clicked.connect(self.record_webcam)
        self.stopRecordButton.clicked.connect(self.stop_record_webcam)

        self.label.mousePressEvent = self.getPos
        self.biglabel.mousePressEvent = self.setFixationPoint

        self.produceBackprojectedButton.clicked.connect(self.backprojectHighlighted)
        self.produceCorticalButton.clicked.connect(self.corticalHighlighted)

        # Create and load retina
        R = Retina()
        R.info()
        R.loadLoc(join(datadir, "retinas", "ret50k_loc.pkl"))
        R.loadCoeff(join(datadir, "retinas", "ret50k_coeff.pkl"))
        self.R = R


        # Create and prepare cortex
        C = Cortex()
        lp = join(datadir, "cortices", "50k_Lloc_tight.pkl")
        rp = join(datadir, "cortices", "50k_Rloc_tight.pkl")
        C.loadLocs(lp, rp)
        C.loadCoeffs(join(datadir, "cortices", "50k_Lcoeff_tight.pkl"), join(datadir, "cortices", "50k_Rcoeff_tight.pkl"))
        self.C = C

    #prepare retina by loading in a new fixation point, is called whenever fixation point changes
    def prepare_retina(self):
        if self.image is not None:
            print('enters here')
            #x = self.image.shape[1] / (480/self.xspinBox.value())
            #y = self.image.shape[0] / (360/self.yspinBox.value())

            pprint(self.image.shape)

            x = (self.xspinBox.value() / 480) * self.image.shape[1]
            y = (self.yspinBox.value() / 360) * self.image.shape[0]

            print(x)
            print(self.image.shape[1])
            print(y)
            print(self.image.shape[0])

            fixation = (y, x)
            self.R.prepare(self.image.shape, fixation)

    #returns the backprojected image given an imagevector and a (new set of fixation points)
    def getBackprojection(self, vector, x=None, y=None):
        self.V = self.R.sample(vector, (y, x))
        return self.R.backproject_tight_last()


    #returns the cortical image given an imagevector and a (new set of fixation points)
    def getCortical(self, vector, x=None, y=None):
        self.V = self.R.sample(vector, (y, x))
        return self.C.cort_img(self.V)

    '''
    # Deprecated implementation of the webcam module, imported from the webcam recorder proof of concept
    
    def runWebCam(self, event):
        """Starts video player in webcam mode and deactivates relevant buttons"""
        if event:
            print('s1')
            self.currentFile = 0
            self.startVideoPlayer(webcammode=True)
            self.browseButton.setDisabled(True)
            self.browseFolderButton.setDisabled(True)
        else:
            if self.currentFrames:
                print('s2')
                self.startVideoPlayer()
            else:
                print('s3')
                self.videoPlayer = None
            self.browseButton.setDisabled(False)
            self.browseFolderButton.setDisabled(False)
    '''

    """
    Function called when the biglabel object on the main tab is clicked with the checkbox ticked,
    sets the fixation point of the ImageVector object displayed to the point relative to the dimensions of the label and
    the size of the object.
    """
    def setFixationPoint(self, event):
        if self.fixationMouseButton.isChecked():
            try:
                num = self.getCurrentFrameNum() - 1
                self.currentFrames[num].fixationy = (event.pos().y()/720)* self.currentFrames[num]._vector.shape[0]
                self.currentFrames[num].fixationx = (event.pos().x()/1280)* self.currentFrames[num]._vector.shape[1]
                self.saveMetaData(num)
                self.displayMetaData(num)
            except Exception as e:
                print("no frame selected: " + str(e))

    """
    Function that changes the value of the spinBox objects on the capture tab when the webcam feed is clicked
    """
    def getPos(self, event):
        print('label clicked')
        self.xspinBox.setValue(event.pos().x())
        self.yspinBox.setValue(event.pos().y())

    def runWebCam(self, event):

        self.startButton.setEnabled(False)
        """Starts video player in webcam mode and deactivates relevant buttons"""
        if event:
            self.currentFile = 0
            self.startVideoPlayer(webcammode=True)
            self.browseButton.setDisabled(True)
            self.browseFolderButton.setDisabled(True)
        else:
            if self.currentFrames:
                self.startVideoPlayer()
            else:
                self.videoPlayer = None
            self.browseButton.setDisabled(False)
            self.browseFolderButton.setDisabled(False)

        self.images = []

        self.capture = cv2.VideoCapture(0)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, float(self.height))
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, float(self.width))

        self.pauseButton.setEnabled(True)
        self.recordButton.setEnabled(True)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)

        # this sets the fps of the camera (fps = 1000/fps), not greater than max fps of the camera
        self.timer.start(self.framerate)


    """
    Fetches the next frame from the webcam feed, produces the retina images and produces a new ImageVector object to
    save them if necessary, then calls the display method
    """
    def update_frame(self):
        self.cap.start()
        ret, self.image = self.cap.read()

        if not self.setup:
            self.prepare_retina()
            self.setup = True

        fixation = (self.yspinBox.value() * self.image.shape[0] / 360, self.xspinBox.value() * self.image.shape[1] / 480)

        if ret is True:
            #self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
            pprint(self.image.shape)
            self.V = self.R.sample(self.image, fixation)
            backprojected_img = self.R.backproject_tight_last()
            cortical_img = self.C.cort_img(self.V)

            self.displayImage(self.image, cortical=cortical_img, backp=backprojected_img)

        else:
            self.displayImage(self.image, 1)

        if self.flagRecord:
            print('RECORDING...)')

        flag = False
        if self.flagRecord:
            if self.checkUnedited.isChecked():
                flag = True
                self.currentFrames.append(ImageVector(self.image, name='webcam' + str(len(self.frames) + len(self.currentFrames) + 1), vectortype='raw', fixationx=self.xspinBox.value(), fixationy=self.yspinBox.value(), framenum=len(self.currentFrames)+1))
            if self.checkBackprojected.isChecked():
                flag = True
                self.currentFrames.append(ImageVector(backprojected_img, name='webcam_backprojected' +  str(len(self.frames) + len(self.currentFrames) + 1), vectortype='backprojection', fixationx=self.xspinBox.value(), fixationy=self.yspinBox.value(), framenum=len(self.currentFrames) + 1))
            if self.checkRetina.isChecked():
                flag = True
                self.currentFrames.append(ImageVector(cortical_img, name='webcam_cortical' +  str(len(self.frames) + len(self.currentFrames) + 1), vectortype='cortical', fixationx=self.xspinBox.value(), fixationy=self.yspinBox.value(), framenum=len(self.currentFrames) + 1))
            if flag:
                self.fillGallery()
                self.displayMetaData()

    """Stops the webcam feed and the generation of retina images"""
    def stop_webcam(self):

        self.timer.stop()
        self.cap.stop()

        self.startButton.setEnabled(True)
        self.recordButton.setEnabled(False)
        self.pauseButton.setEnabled(False)
        self.stopRecordButton.setEnabled(False)

        print(len(self.images))
        # for x in self.images:
        #    pprint(x)
        #    print("\n")

    """Takes the webcam feed and retina images and displays them on the capture tab"""
    def displayImage(self, img, window=1, backp=None, cortical=None):

        qformat = QImage.Format_Indexed8
        if len(img.shape) == 3:  # [0]=rows, [1]=cols, [2]=channels (colours)
            if img.shape[2] == 4:
                qformat = QImage.Format_RGBA8888
            else:
                qformat = QImage.Format_RGB888

        outImage = qimage2ndarray.array2qimage(img)
        outImage = outImage.rgbSwapped()
        #outImage = QImage(img, img.shape[1], img.shape[0], img.strides[0], qformat)

        if backp is not None:
            outFocalImage = qimage2ndarray.array2qimage(backp)
            outFocalImage = outFocalImage.rgbSwapped()

        if cortical is not None:

            outCorticalImage = qimage2ndarray.array2qimage(cortical)
            outCorticalImage = outCorticalImage.rgbSwapped()
            #outCorticalImage = ip.convertToPixmap(cv2.cvtColor(cortical, cv2.COLOR_BGR2RGB), 480, 360)


        if window == 1:
            self.label.setPixmap(QPixmap.fromImage(outImage))
            self.label.setScaledContents(True)

            if backp is not None:
                self.focalLabel.setPixmap(QPixmap.fromImage(outFocalImage))
                self.focalLabel.setScaledContents(True)

            if cortical is not None:
                self.corticalLabel.setPixmap(QPixmap.fromImage(outCorticalImage))
                self.corticalLabel.setScaledContents(True)

    """Clears memory when record is pressed and activates/disables relative buttons"""
    def record_webcam(self):
        self.currentFrames = []
        self.flagRecord = True
        self.stopRecordButton.setEnabled(True)
        self.recordButton.setEnabled(False)
        self.pauseButton.setEnabled(False)
        self.labelRecording.show()

    """Saves captured retina images and nd activates/disables relative buttons"""
    def stop_record_webcam(self):
        self.frames.extend(self.currentFrames)
        self.currentFrames = []
        self.flagRecord = False
        self.stopRecordButton.setEnabled(False)
        self.recordButton.setEnabled(True)
        self.pauseButton.setEnabled(True)
        self.labelRecording.hide()
        print(len(self.frames))
        self.setCurrentFrames(self.frames)

    """Configures file browser parameters"""
    def openFileNameDialog(self):
        options = QFileDialog.Options()
        self.dialog = QFileDialog.DontUseNativeDialog
        options |= self.dialog
        fileName, _ = QFileDialog.getOpenFileName(self,"Open file", "",
            "All Files (*);;mp4 Files (*.mp4);;avi Files (*.avi);;jpeg Files (*.jpg);;csv Files (*.csv);;HDF5 Files(*.h5)",
            options=options)
        # only continue if a name has been specified
        if fileName:
            self.openFile(fileName)

    """Opens the file browser"""
    def openFile(self,filename):
        filetype = filename.split('.')[-1]
        self.generateButton.setText("Loading...")
        self.infoLabel.setText("File opened: " + filename)
        print("Opening " + filename)
        # start the video player or start a worker thread depending on file type
        if filetype in self.videofiletypes:
            self.currentFile = Video(filepath=filename,colortype="rgb")
            self.generateButton.setText("Generate images from video")
            self.generateButton.setDisabled(False)
            self.startVideoPlayer()
        elif filetype in self.metadatatypes:
            self.metafilename = filename
            self.startWorker(ip.loadCsv,self.setCurrentFrames,self.fillGallery,
                self.metafilename,self.currentFrames)
        elif filetype == 'pkl':
            self.currentFile = filename
            self.startWorker(ip.loadPickle,self.setCurrentFrames,self.fillGallery,
                self.currentFile)
        elif filetype == 'h5':
            self.currentFile = filename
            self.startWorker(ip.loadhdf5,self.setCurrentFrames,self.fillGallery,
                self.currentFile,self.currentFrames)
        else:
            # invalid file type selected
            self.showWarning('FileType')

    def openFolderDialog(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        datadir = QFileDialog.getExistingDirectory(self, "Open folder")
        # only continue if a folder was chosen
        if datadir:
            print("Directory opened:" + datadir)
            self.currentDir = datadir
            self.startWorker(ip.createImageVectorsFromFolder,self.setCurrentFrames,
                self.fillGallery,self.currentDir)
            self.infoLabel.setText("Folder opened: "+ self.currentDir)
            self.generateButton.setText("Loading...")
            self.generateButton.setDisabled(True)

    def startWorker(self,func,resultfunc=None,finishedfunc=None,*args):
        """Instantiates a Worker, adds it to the pool and starts it

        Parameters
        ----------
        func : function of task to be performed
        resultfunc : function to receive return value
        finishedfunc : function to be called on finished signals
        args : arguments to be passed with the task function
        """
        worker = Worker(func,*args)
        if resultfunc: worker.signals.result.connect(resultfunc)
        if finishedfunc: worker.signals.finished.connect(finishedfunc)
        worker.signals.error.connect(self.showWarning)
        self.threadpool.start(worker)

    def loadPickle(self):
        return utils.loadPickle(self.currentFile)

    def displayMetaData(self,framenum=0):
        """Gets metadata for selected frame and adds it to the displayed model

        Parameters
        ----------
        framenum : index of object whose metadata is required
        """

        # only proceed if frames are available and valid framnum is chosen
        if self.currentFrames and framenum >= 0 and framenum < len(self.currentFrames):
            # instantiate a new QStandardItemModel which will hold the data
            self.metadatamodel = PyQt5.QtGui.QStandardItemModel(self)
            currentframe = self.currentFrames[int(framenum)]
            # get attribute names of object in question
            labels = dir(self.currentFrames[0])
            items = []
            values = []
            for label in labels:
                # add attribute names to first column
                item = QStandardItem(label.replace("_",""))
                item.setFlags(PyQt5.QtCore.Qt.ItemIsEnabled)
                items.append(item)
                # add attribute values to second column
                value = QStandardItem(str(getattr(currentframe, label)))
                values.append(value)

            self.metadatamodel.appendColumn(items)
            self.metadatamodel.appendColumn(values)
            # set model to that of the metadata table gui element
            self.metadata.setModel(self.metadatamodel)
            # set the image displayed in the large display to the current image


            #height, width, channel = currentframe._vector.shape
            #bytesPerLine = 3 * width
            #qImg = QImage(currentframe._vector, width, height, bytesPerLine, QImage.Format_RGB888).rgbSwapped()
            #pixmap = QPixmap.fromImage(qImg)
            #smaller_pixmap = pixmap.scaled(1280, 720, Qt.KeepAspectRatio, Qt.FastTransformation)
            #self.biglabel.setPixmap(smaller_pixmap)
            #self.biglabel.setImage(qImg)

            image = qimage2ndarray.array2qimage(currentframe._vector)
            #image = image.scaled(1280, 720, Qt.KeepAspectRatio)
            # ...and then convert to QPixmap
            pixmap = QPixmap.fromImage(image.rgbSwapped())
            self.biglabel.setPixmap(pixmap)
            self.biglabel.setScaledContents(True)
            #self.biglabel.setPixmap(ip.convertToPixmap(cv2.cvtColor(currentframe._vector, cv2.COLOR_BGR2RGB),1280,720))
            # add this frame to the list of highlighted images


            if framenum not in self.highlightedframes:
                self.highlightedframes.append(int(framenum))

    def removeHighlighted(self,framenum):
        """Removes selected frame from list of highlighted frames

        Parameters
        ----------
        framenum : index of object which is no longer highlighted/selected
        """
        #print("highlighted frames: " + ' '.join(str(e) for e in self.highlightedframes))
        if framenum in self.highlightedframes:
            #print("removing " + str(framenum))
            self.highlightedframes.remove(framenum)
        #print("remaining frames: " + ' '.join(str(e) for e in self.highlightedframes))

    def saveMetaData(self,framenum):
        """Saves metadata for selected frame, retrieving from metadata table

        Parameters
        ----------
        framenum : index of object whose metadata must be saved
        """
        if self.currentFrames:
            # if we're on the main tab, this frame is the target
            if self.maintabWidget.currentIndex() == 2:
                targetframes = [self.currentFrames[i] for i in self.highlightedframes]
            # if we're on the gallery page, the target is all the highlighted frames
            else:
                targetframes = [f for f in self.currentFrames if f.framenum == self.getCurrentFrameNum()]
            for targetframe in targetframes:
                # scan the metadata table
                for i in range(self.metadatamodel.rowCount()):
                    field = str(self.metadatamodel.item(i,0).text())
                    value = str(self.metadatamodel.item(i,1).text())
                    # only store changes to the label in the current version
                    if field != 'name' or 'label' or 'retinatype':
                        pass
                    else:
                        setattr(targetframe, field, value)

    def getVideoFrames(self):
        """Requests the Video object to break itself into frames"""
        if self.currentFile:
            # this is a lengthy process, use a worker
            self.startWorker(self.currentFile.getFramesImageVectors,self.setCurrentFrames,self.fillGallery)
            self.generateButton.setText("Generating...")
            self.verticalSlider_3.valueChanged.connect(self.fillGallery)
            self.generateButton.setDisabled(True)

    def setCurrentFrames(self, frames):
        """Sets the global list of current frames to the argument given

        Connected to the result signal of Worker objects so we can receive
         the returned frames

         Parameters
         -------
         frames : list of objects that must be set as current frames
         """
        self.currentFrames.extend(frames)
        self.verticalSlider_3.valueChanged.connect(self.fillGallery)
        self.generateButton.setText("Done!")
        numframes = len(self.currentFrames)
        # adjust the range of the gallery slider depending on length of list
        self.verticalSlider_3.setRange(0,numframes/16)
        # reset the video player when new things are imported
        self.startVideoPlayer()

    def getCurrentFrameNum(self):
        """Searches the metadata table for the framenum whose data is currently displayed

        Returns
        -------
        targetframe : frame number found in metadata table
        """
        for i in range(self.metadatamodel.rowCount()):
            if self.metadatamodel.item(i,0).text() == 'framenum':
                targetframe = int(self.metadatamodel.item(i,1).text())
        return targetframe

    def selectAll(self):
        """'Selects' all the items in the gallery by adding them to the highlighted frames list"""
        if self.currentFrames:
            for label in self.labels:
                # highlight all the gallery items
                if label.pixmap:
                    label.makeHighlighted()
            self.maintabWidget.setCurrentIndex(2)
            self.highlightedframes = range(len(self.currentFrames))

    def deleteFrame(self):
        """Deletes selected frames from the list"""
        if self.currentFrames:
            # if we're on the gallery tab, the target is all highlighted frames
            if self.maintabWidget.currentIndex() == 2:
                print("Deleting frames: " + ' '.join(str(e) for e in self.highlightedframes))
                targetframes = [self.currentFrames[i] for i in self.highlightedframes]
            # if we're on the main tab, the target is the currently displayed frame
            else:
                targetframes = [f for f in self.currentFrames if f.framenum == self.getCurrentFrameNum()]
            self.currentFrames = [f for f in self.currentFrames if f not in targetframes]
            # update after changes
            self.fillGallery()
            self.updateVideoPlayer()

    def closeFile(self):
        """Closes the current project by deleting the current frames and file reference"""
        if self.currentFile: self.currentFile = None
        if self.currentFrames:
            self.currentFrames = []
            self.fillGallery()
            self.updateVideoPlayer()

    def updateVideoPlayer(self):
        """Updates the video player after changes to the current frames have been made"""
        self.videoPlayer.frames = self.currentFrames
        self.videoPlayer.maxFrames = len(self.currentFrames) - 1
        #self.videoPlayer.framePos = 0
        #self.videoPlayer.nextFrame()
        self.videoPlayer.setCurrent()

    def saveFileDialog(self):
        fileName, _ = QFileDialog.getSaveFileName(self,"Save file","","csv (*.csv);;HDF5 (*.h5);;pickle (*.pkl)")#, options=options
        filetype = fileName.split(".")[-1]
        # only proceed if a file name was chosen
        #extension
        pprint(filetype)
        #
        if fileName and self.currentFrames:
            self.exportfilename = fileName
            # start different workers depending on file type
            if filetype == 'pkl':
                utils.writePickle(fileName, self.currentFrames)
            elif filetype == 'h5' and isinstance(self.currentFrames[0], ImageVector):
                self.startWorker(ip.saveHDF5,None,self.generateButton.setText("Done!"),
                    self.exportfilename,self.currentFrames)
                self.generateButton.setText("Saving to HDF5...")
            elif filetype == 'csv' and isinstance(self.currentFrames[0], ImageVector):
                self.startWorker(ip.saveCSV,None,None,self.exportfilename,self.currentFrames)
                self.generateButton.setText("Saving to CSV...")
            else:
                # invalid file type specified
                self.showWarning('FileType')

    def setRetinaEnabled(self, event):
        """Sets the boolean for retina use and (de)activates relevant buttons"""
        if event:
            self.isRetinaEnabled = True
            self.browseButton.setDisabled(True)
            self.browseFolderButton.setDisabled(True)
            self.actionFile.setDisabled(True)
            self.actionFolder.setDisabled(True)
        else:
            self.isRetinaEnabled = False
            self.browseButton.setDisabled(False)
            self.browseFolderButton.setDisabled(False)
            self.actionFile.setDisabled(False)
            self.actionFolder.setDisabled(False)

    def startVideoPlayer(self, webcammode=False):
        """Instantiates and initializes the video player and connects relevant buttons"""
        self.videoPlayer = VideoPlayer(self.currentFile,self.isRetinaEnabled,self,webcammode)
        self.pauseButton.clicked.connect(self.videoPlayer.pause)
        self.startButton.clicked.connect(self.videoPlayer.start)
        self.pauseButton_2.clicked.connect(self.videoPlayer.pause)
        self.startButton_2.clicked.connect(self.videoPlayer.start)
        # don't connect all buttons if in webcam mode
        if not webcammode:
            self.skipBackButton.clicked.connect(self.videoPlayer.skipBck)
            self.skipForwardButton.clicked.connect(self.videoPlayer.skipFwd)
            self.scrubSlider.valueChanged.connect(self.sendFramePos)
            self.skipBackButton_2.clicked.connect(self.videoPlayer.skipBck)
            self.skipForwardButton_2.clicked.connect(self.videoPlayer.skipFwd)
            self.scrubSlider_2.valueChanged.connect(self.sendFramePos)
            self.skipBackButton_4.clicked.connect(self.videoPlayer.skipToFirst)
            self.skipForwardButton_4.clicked.connect(self.videoPlayer.skipToLast)

    def sendFramePos(self):
        """Passes the frame index to the video player"""
        framePos = self.scrubSlider.value()
        self.videoPlayer.skip(framePos)

    def fillGallery(self):
        """Updates the gallery items and frame number labels"""
        for i in range(len(self.labels)):
            if i < len(self.currentFrames):
                tempindex = i + (16 * self.verticalSlider_3.value())
                pprint(str(tempindex) + 'of' + str(len(self.currentFrames)))
                if tempindex < len(self.currentFrames):
                    currentframe = self.currentFrames[tempindex]
                    try:
                        self.labels[i].setPixmap(QPixmap.fromImage(qimage2ndarray.array2qimage(cv2.cvtColor(currentframe._vector, cv2.COLOR_BGR2RGB))))
                        self.labels[i].setScaledContents(True)
                        self.labels[i].setIndex(tempindex)
                        self.numbers[i].display(currentframe.framenum)
                    except:
                        pprint('error')
                else:
                    self.labels[i].clear()
                    self.labels[i].setIndex(-1)
                    self.numbers[i].display(0)
            else:
                self.labels[i].clear()
                self.labels[i].setIndex(-1)
                self.numbers[i].display(0)
            self.labels[i].notHighlighted()
        self.highlightedframes = []

    def showWarning(self,error):
        """Helps generate message boxes with relevant messages"""
        if isinstance(error, tuple):
            messagekey = str(error[1])
        else:
            messagekey = error
        messages = {
        'exceptions.IndexError' : 'There is an inequal number of images and metadata records',
        '1L' : 'There was a problem with the format of the metadata',
        'NoFrames' : 'Please load images before loading metadata',
        'HDF5Format' : 'There was a problem with the format of the HDF5 file',
        'CSVFormat' : 'There was a problem with the format of the CSV file',
        'FileType' : 'File type not supported',
        'InvalidFrameNum' : 'Frame number must be an integer'
        }
        errormessage = QMessageBox(parent=None)
        errormessage.setStandardButtons(QMessageBox.Ok)
        errormessage.setWindowTitle('Warning')
        errormessage.setIcon(QMessageBox.Warning)
        errormessage.setText(messages[messagekey])
        errormessage.exec_()

    def backprojectHighlighted(self):
        start = time.time()
        for frame in self.highlightedframes:
            pprint(frame)
            if self.currentFrames[frame].vectortype != "backprojected" and self.currentFrames[frame].fixationx is not None and self.currentFrames[frame].fixationy is not None and self.currentFrames[frame]._vector is not None:
                x = self.currentFrames[frame].fixationx
                y = self.currentFrames[frame].fixationy
                fixation = (y, x)
                self.currentFrames[frame]._vector = cv2.cvtColor(self.currentFrames[frame]._vector, cv2.COLOR_BGR2GRAY)
                self.R.prepare(self.currentFrames[frame]._vector.shape, fixation)
                self.V = self.R.sample(self.currentFrames[frame]._vector, fixation)
                backprojected_img = self.R.backproject_tight_last()
                self.currentFrames.append(ImageVector(backprojected_img, name=(self.currentFrames[frame].name + '_backprojection'), vectortype='backprojection', fixationx=self.currentFrames[frame].fixationx, fixationy=self.currentFrames[frame].fixationy, framenum=len(self.currentFrames) + 1, label=self.currentFrames[frame].label))
        end = time.time()
        # inform user of operation time for diagnostics
        print("This operation required " + str(end - start) + " seconds.")
        self.fillGallery()
        self.displayMetaData()



    def corticalHighlighted(self):
        start = time.time()
        for frame in self.highlightedframes:
            pprint(frame)
            if self.currentFrames[frame].vectortype != "cortical" and self.currentFrames[frame].fixationx is not None and self.currentFrames[frame].fixationy is not None and self.currentFrames[frame]._vector is not None:
                x = self.currentFrames[frame].fixationx
                y = self.currentFrames[frame].fixationy
                fixation = (int(y), int(x))
                #self.currentFrames[frame]._vector = cv2.cvtColor(self.currentFrames[frame]._vector, cv2.COLOR_BGR2GRAY)
                self.R.prepare(self.currentFrames[frame]._vector.shape, fixation)
                self.V = self.R.sample(self.currentFrames[frame]._vector, fixation)
                cortical_img = self.C.cort_img(self.V)
                self.currentFrames.append(
                    ImageVector(cortical_img, name=(self.currentFrames[frame].name + '_cortical'), vectortype='cortical', fixationx=self.currentFrames[frame].fixationx, fixationy=self.currentFrames[frame].fixationy, framenum=len(self.currentFrames) + 1, label=self.currentFrames[frame].label))
        end = time.time()
        # inform user of operation time for diagnostics
        print("This operation required " + str(end - start) + " seconds.")
        self.fillGallery()
        self.displayMetaData()

    def closeApp(self):
        sys.exit()


def main():
    app = QApplication(sys.argv)
    form = DMApp()
    app.setStyle('Fusion')
    form.show()
    app.exec_()

if __name__ == '__main__':
    main()
