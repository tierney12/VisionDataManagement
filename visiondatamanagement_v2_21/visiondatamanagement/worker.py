"""
Created on 8/7/2018 14:36

Worker class employed when tasks need to be performed outside the event thread.
Can be given any function, and up to two functions can be signaled on completion,
one with the return value.

@author: Connor Fulton
"""

import sys, traceback
import os
import time

from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *

class Workersignals(QObject):
    """Helps manage the signals of a worker object"""
    finished = pyqtSignal()
    error = pyqtSignal(tuple)
    result = pyqtSignal(object)
    progress = pyqtSignal(int)

class Worker(QRunnable):
    """Executes task specified during instantiation and produces signals about status"""
    def __init__(self, fn, *args, **kwargs):
        super(Worker, self).__init__()
        self.fn = fn
        self.args = args
        self.kwargs = kwargs
        self.signals = Workersignals()
        self.kwargs['progress_callback'] = self.signals.progress

    @pyqtSlot()
    def run(self):
        """Runs the task and reports exceptions or result"""
        try:
            start = time.time()
            result = self.fn(*self.args)
        except:
            traceback.print_exc()
            exctype, value = sys.exc_info()[:2]
            # send the exception info back to the error slot
            self.signals.error.emit((exctype, value, traceback.format_exc()))
        else:
            # if operation was successful, send the results back
            self.signals.result.emit(result)
            self.signals.finished.emit()
        finally:
            end = time.time()
            # inform user of operation time for diagnostics
            print("This operation required " + str(end - start) + " seconds.")

