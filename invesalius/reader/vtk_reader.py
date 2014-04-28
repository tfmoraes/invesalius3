import os
import multiprocessing
import tempfile

import vtk

def ReadVTK(filename):
    r = vtk.vtkXMLImageDataReader()
    r.SetFileName(filename)
    r.Update()

    return r.GetOutput()

def ReadDirectory(dir_):
    """ 
    Looking for analyze files in the given directory
    """
    imagedata = None
    for root, sub_folders, files in os.walk(dir_):
        for file in files:
            if file.split(".")[-1] == "vti":
                filename = os.path.join(root,file)
                imagedata = ReadVTK(filename)
                return imagedata
    return imagedata
