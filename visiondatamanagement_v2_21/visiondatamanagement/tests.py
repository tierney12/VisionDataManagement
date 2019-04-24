from __future__ import print_function
import unittest
import sys
import os
import main
import csv
import timeit
import gc
import numpy as np
from unittest import TestCase
from main import DMApp


from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
import processing as ip
from model import ImageVector
from retinavision.retina import Retina
from retinavision.cortex import Cortex
from retinavision import datadir, utils
import cv2


@unittest.skip("Skipping functional tests")
class AppTests(TestCase):

    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

    def setUp(self):
        self.app = QApplication(sys.argv)
        self.form = DMApp()

    @unittest.skip("Skipping HDF5 consistency test")
    def test_hdf5_consistency(self):
        print("Loading HDF5 test file...")
        testframes = ip.loadhdf5(os.path.join("testdata","testfilesmall.h5"),None)
        print("Saving HDF5 test file unchanged...")
        ip.saveHDF5(os.path.join("testdata","testfileoutput.h5"),testframes)

        # differences between the files can be observed using h5diff, a tool of the HDF5
        # package from the command line. Attempts to find a programmatic method of comparing
        # files that isn't extremely time consuming have been unsuccessful, but I will
        # continute looking until delivery.

    @unittest.skip("skipping CSV consistency test")
    def test_csv_consistency(self):
        print("Loading CSV test file...")
        testframes = ip.loadCsv(os.path.join("testdata","extrasmalltestfile.csv"),None)
        print("Saving CSV test file unchanged...")
        ip.saveCSV(os.path.join("testdata","csvconsistencyoutput.csv"),testframes)


        # differences between the files must be observed manually at the moment. Attempts
        # to find a programmatic method of comparing files that isn't extremely time consuming
        # have been unsuccessful, but I will continute looking until delivery.

#        with open(os.path.join("testdata","new.csv"), 'r') as filein, open(os.path.join("testdata","newoutput.csv"), 'r') as fileout:
#            f1 = filein.readlines()
#            f2 = fileout.readlines()
#
#        with open(os.path.join("testdata","diff.csv"), 'w') as diffFile:
#            for line in f2:
#                if line not in f1:
#                    diffFile.write(line)

#        filein = frozenset(tuple(row) for row in csv.reader(open(os.path.join("testdata","new.csv"), 'r'), delimiter=' '))
#        fileout = frozenset(tuple(row) for row in csv.reader(open(os.path.join("testdata","newoutput.csv"), 'r'), delimiter=' '))

#        added = [" ".join(row) for row in fileout - filein]
#        removed = [" ".join(row) for row in filein - fileout]
#        print("Differences between the input and output files have been written to diff.csv")

    def test_prohibited_file_types(self):
        # reference to valid hdf5 file is stored
        print("Opening HDF5 test file...")
        filename = os.path.join("testdata","testfile.h5")
        self.form.openFile(filename)
        self.assertEquals(self.form.currentFile, filename, 'Valid file reference not stored')
        print('Valid file reference stored')

        self.form.currentFile = None # manually delete to avoid dependency on closeFile()

        # reference to invalid doc file is not stored
        print("Opening invalid file format (.doc)...")
        filename = os.path.join("testdata","testfile.doc")
        self.form.openFile(filename)
        self.assertIsNone(self.form.currentFile, 'Invalid file reference stored')
        print('Invalid file reference not stored')

    def test_close_file(self):
        print("Closing test file...")
        filename = os.path.join("testdata","testfile.h5")
        self.form.openFile(filename)
        self.form.closeFile()
        self.assertIsNone(self.form.currentFile, 'File not closed, reference still exists')
        print('File successfully closed')

    def test_hdf5_format_restricted(self):
        print("Loading HDF5 test file with no imagevectors...")
        self.assertRaises(Exception,ip.loadhdf5,os.path.join("testdata","invalidtestfile.h5"))
        print("Exception raised on loading of hdf5 file with no imagevectors")

    def test_HDF5_file_with_partial_metadata_rejected(self):
        print("Loading HDF5 file with incomplete metadata...")
        self.assertRaises(Exception,ip.loadhdf5,os.path.join("testdata","partialmetadatafile.h5"),None)
        print("Exception raised on loading of HDF5 file with only partial metadata")

    def test_HDF5_file_with_unequal_dataset_lengths_rejected(self):
        print("Loading HDF5 file with unequal length data sets...")
        self.assertRaises(Exception,ip.loadhdf5,os.path.join("testdata","unequallengthdatafile.h5"),None)
        print("Exception raised on loading of HDF5 file with unequal dataset lengths")

    def test_CSV_file_with_partial_metadata_rejected(self):
        print("Loading CSV file with incomplete metadata...")
        self.assertRaises(Exception,ip.loadCsv,os.path.join("testdata","partialmetadatafile.csv"),None)
        print("Exception raised on loading of CSV file with only partial metadata")

    def test_CSV_file_with_unequal_dataset_lengths_rejected(self):
        print("Loading CSV file with unequal length data sets...")
        self.assertRaises(Exception,ip.loadCsv,os.path.join("testdata","unequallengthdatafile.csv"),None)
        print("Exception raised on loading of CSV file with unequal dataset lengths")

    def test_loading_metadata_before_imagedata_prohibited(self):
        print("Loading CSV test file before vectors...")
        self.assertRaises(Exception,ip.loadCsv,os.path.join("testdata","novectorstestfile.csv"),None)
        print("Exception raised on loading metadata before vectors")

    def test_metadata_value_type_restricted(self):
        # currently the only metadata type that allows editing is the label,
        # which can be anything. Wider editing possibilities may be added
        pass

    # Test whether an imagevector loaded into the model and saved to file is identical to the original imagevector.
    # Assure that the cortex module still accepts the imagevector and produces the same result. Setup of retina for
    # original imagevector is done without using app functions to avoid dependency on them.
    def test_imagevector_integrity(self):
        print("Testing imagevector integrity over storing in model and storing in file...")
        cap = utils.camopen()
        ret, campic = cap.read()

        R = Retina()
        R.info()
        R.loadLoc(os.path.join(datadir, "retinas", "ret50k_loc.pkl"))
        R.loadCoeff(os.path.join(datadir, "retinas", "ret50k_coeff.pkl"))

        x = campic.shape[1]/2
        y = campic.shape[0]/2
        fixation = (y,x)
        R.prepare(campic.shape, fixation)

        ret, img = cap.read()
        if ret:
            V = R.sample(img, fixation)

        originalvector = ImageVector(V,framenum=1,timestamp="HH:MM:SS.SS",label="dummy",fixationy=y,fixationx=x,retinatype="50k")
        self.assertTrue(np.array_equal(V,originalvector._vector),'Vector has been modified by storage in model')

        vectors = [originalvector]
        ip.saveHDF5(os.path.join("testdata","integritytest.h5"),vectors)
        newvectors = ip.loadhdf5(os.path.join("testdata","integritytest.h5"),None)
        newvector = newvectors[0]
        self.assertTrue(np.array_equal(V,originalvector._vector),'Vector has been modified by storage in HDF5')

        ip.saveCSV(os.path.join("testdata","integritytest.csv"),vectors)
        newvectors = ip.loadCsv(os.path.join("testdata","integritytest.csv"),None)
        newvector = newvectors[0]
        self.assertTrue(np.array_equal(V,originalvector._vector),'Vector has been modified by storage in CSV')

        print("Vector has not been modified by the system")

#    def tearDown(self):
#        pass

#@unittest.skip("Skipping performance tests")
class PerformanceTests(TestCase):

    pass
    # Results are larger than typical for normal use, but more accurate, as timeit
    # module disables garbage collection.

    # Loading a HDF5 file. Generation of backprojections is included in this test,
    # as this is always the procedure. This requires the majority of the execution
    # time.
    @unittest.skip("Skipping HDF5 load performance test")
    def test_hdf5_loading_performance(self):
        # key: sm - small file, lg - large file, nc - nonconcurrent, c - nonconcurrent, vf - varying fixation

        sm_nc_times,sm_c_times,lg_nc_times,lg_c_times = (0 for i in range(4))
        setup = "import gc;gc.enable();import processing as ip; import os"
        runs = 10
        results = open('hdf5_loading_results.txt','w')

        sm_nc_times = timeit.timeit('ip.loadhdf5(os.path.join("testdata","testfilesmall.h5"),None)',setup=setup,number=runs)
        print("Average time to load small hdf5 non-concurrently ("+str(runs)+" runs): " + str(sm_nc_times/float(runs)) + " seconds", file=results)

        #gc.collect()

        #sm_c_times = timeit.timeit('ip.loadhdf5(os.path.join("testdata","testfilesmall.h5"),None,concurrbackproject=True)',setup=setup,number=runs)
        #print("Average time to load small hdf5 concurrently ("+str(runs)+" runs): " + str(sm_c_times/float(runs)) + " seconds", file=results)

        gc.collect()

        lg_nc_times = timeit.timeit('ip.loadhdf5(os.path.join("testdata","testfile.h5"),None)',setup=setup,number=runs)
        print("Average time to load large hdf5 non-concurrently ("+str(runs)+" runs): " + str(lg_nc_times/float(runs)) + " seconds", file=results)

        gc.collect()

        #lg_c_times = timeit.timeit('ip.loadhdf5(os.path.join("testdata","testfile.h5"),None,concurrbackproject=True)',setup=setup,number=runs)
        #print("Average time to load large hdf5 concurrently ("+str(runs)+" runs): " + str(lg_c_times/float(runs)) + " seconds", file=results)

        gc.collect()

        sm_nc_vf_times = timeit.timeit('ip.loadhdf5(os.path.join("testdata","varyingfixationfilesmall.h5"),None)',setup=setup,number=runs)
        print("Average time to load small hdf5 with varying fixation non-concurrently ("+str(runs)+" runs): " + str(sm_nc_vf_times/float(runs)) + " seconds", file=results)

        gc.collect()

        lg_nc_vf_times = timeit.timeit('ip.loadhdf5(os.path.join("testdata","varyingfixationfile.h5"),None)',setup=setup,number=runs)
        print("Average time to load large hdf5 with varying fixation non-concurrently ("+str(runs)+" runs): " + str(lg_nc_vf_times/float(runs)) + " seconds", file=results)

        print("HDF5 loading performance tests complete.")

    @unittest.skip("Skipping CSV load performance test")
    def test_csv_loading_performance(self):
        # key: sm - small file, lg - large file, nc - nonconcurrent, c - nonconcurrent, vf - varying fixation

        sm_nc_times,sm_c_times,lg_nc_times,lg_c_times = (0 for i in range(4))
        setup = "import gc;gc.enable();import processing as ip; import os"
        runs = 10
        results = open('csv_loading_results.txt','w')

        sm_nc_times = timeit.timeit('ip.loadCsv(os.path.join("testdata","testfilesmall.csv"),None)',setup=setup,number=runs)
        print("Average time to load small csv non-concurrently ("+str(runs)+" runs): " + str(sm_nc_times/float(runs)) + " seconds",file=results)

        gc.collect()

        #sm_c_times = timeit.timeit('ip.loadCsv(os.path.join("testdata","testfilesmall.csv"),None,concurrbackproject=True)',setup=setup,number=runs)
        #print("Average time to load small csv concurrently ("+str(runs)+" runs): " + str(sm_c_times/float(runs)) + " seconds",file=results)

        #gc.collect()

        lg_nc_times = timeit.timeit('ip.loadCsv(os.path.join("testdata","testfile.csv"),None)',setup=setup,number=runs)
        print("Average time to load large csv non-concurrently ("+str(runs)+" runs): " + str(lg_nc_times/float(runs)) + " seconds",file=results)

        gc.collect()

        #lg_c_times = timeit.timeit('ip.loadCsv(os.path.join("testdata","testfile.csv"),None,concurrbackproject=True)',setup=setup,number=runs)
        #print("Average time to load large csv concurrently ("+str(runs)+" runs): " + str(lg_c_times/float(runs)) + " seconds",file=results)

        #gc.collect()

        sm_nc_vf_times = timeit.timeit('ip.loadCsv(os.path.join("testdata","varyingfixationfilesmall.csv"),None)',setup=setup,number=runs)
        print("Average time to load small csv with varying fixation non-concurrently ("+str(runs)+" runs): " + str(sm_nc_vf_times/float(runs)) + " seconds",file=results)

        gc.collect()

        lg_nc_vf_times = timeit.timeit('ip.loadCsv(os.path.join("testdata","varyingfixationfile.csv"),None)',setup=setup,number=runs)
        print("Average time to load large csv with varying fixation non-concurrently ("+str(runs)+" runs): " + str(lg_nc_vf_times/float(runs)) + " seconds",file=results)

        print("CSV loading performance tests complete.")

    @unittest.skip("Skipping HDF5 saving performance test")
    def test_hdf5_saving_performance(self):
        # key: sm - small file, lg - large file

        sm_times,lg_times = (0 for i in range(2))
        setup = "import gc;gc.enable();import processing as ip; import os"
        runs = 10
        results = open('hdf5_saving_results.txt','w')

        #testframes = ip.loadhdf5(os.path.join("testdata","testfilesmall.h5"),None)
        sm_times = timeit.timeit('ip.saveHDF5(os.path.join("testdata","savetestfilesmall.h5"),ip.loadhdf5(os.path.join("testdata","testfilesmall.h5"),None))',setup=setup,number=runs)
        print("Average time to save small hdf5 ("+str(runs)+" runs): " + str(sm_times/float(runs)) + " seconds", file=results)

        gc.collect()

        #testframes = ip.loadhdf5(os.path.join("testdata","testfile.h5"),None)
        lg_times = timeit.timeit('ip.saveHDF5(os.path.join("testdata","savetestfile.h5"),ip.loadhdf5(os.path.join("testdata","testfile.h5"),None))',setup=setup,number=runs)
        print("Average time to save large hdf5 ("+str(runs)+" runs): " + str(lg_times/float(runs)) + " seconds", file=results)

        gc.collect()

        print("HDF5 saving performance tests complete.")

    #@unittest.skip("Skipping HDF5 saving performance test")
    def test_csv_saving_performance(self):
        # key: sm - small file, lg - large file

        sm_times,lg_imes = (0 for i in range(2))
        setup = "import gc;gc.enable();import processing as ip; import os"
        runs = 10
        results = open('csv_saving_results.txt','w')

        #testframes = ip.loadCsv(os.path.join("testdata","testfilesmall.csv"),None)
        sm_times = timeit.timeit('ip.saveCSV(os.path.join("testdata","savetestfilesmall.csv"),ip.loadCsv(os.path.join("testdata","testfilesmall.csv"),None))',setup=setup,number=runs)
        print("Average time to save small CSV ("+str(runs)+" runs): " + str(sm_times/float(runs)) + " seconds", file=results)

        gc.collect()

        #testframes = ip.loadCsv(os.path.join("testdata","testfile.csv"),None)
        lg_times = timeit.timeit('ip.saveCSV(os.path.join("testdata","savetestfile.csv"),ip.loadCsv(os.path.join("testdata","testfile.csv"),None))',setup=setup,number=runs)
        print("Average time to save large CSV ("+str(runs)+" runs): " + str(lg_times/float(runs)) + " seconds", file=results)

        gc.collect()

        print("CSV saving performance tests complete.")

if __name__ == '__main__':
    unittest.main(exit=False)
