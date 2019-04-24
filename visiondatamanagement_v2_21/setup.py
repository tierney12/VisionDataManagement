import os
import sys
from setuptools import setup
from setuptools.command.install import install


class PostInstallCommand(install):
    """Tasks to perform after installation"""
    def run(self):
        if os.name == 'nt':
            try:
                sys.path.index(os.path.join(os.environ["ProgramFiles"],"NVIDIA CORPORATION","NVSMI"))
            except ValueError:
                sys.path.append(os.path.join(os.environ["ProgramFiles"],"NVIDIA CORPORATION","NVSMI"))
        install.run(self)
        
def read(filename):
    return open(os.path.join(os.path.dirname(__file__), filename)).read()

setup(
    name="visiondatamanagement",
    version="2.21",
    author="Sean P. Tierney",
    author_email="seantierney13@btinternet.com",
    description="A tool for the genertion and management of data produced by Piotr's software retina system.",
    packages=["visiondatamanagement"],
    classifiers=(
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ),
    long_description=read('README.txt'),
    install_requires=[read(os.path.join('visiondatamanagement', 'requirements.txt'))],
    dependency_links=[
    "git+ssh://git@github.com/Pozimek/RetinaVision@1693fbcaad0813a0bc8937b6dd4288cbdd273e4c#egg=retinavision-0.9",
    "git://github.com/pyqt/python-qt5#egg=python-qt5",
    ],
    cmdclass={
        'install': PostInstallCommand,
    },
)