README
=========

Dependencies:
-------------
Please install the following libraries for python 2.7:

- scipy
- numpy
- pandas (just for pretty tables...can be removed)
- matplotlib

All of these come with anaconda's python distribution. On Chewie (Troy's linux workstation), type **pyconda** to launch anaconda's python distribution. Otherwise use python (also has packages installed). If anaconda accelerate is installed, MKL will be used for linear algebra subroutines.

Can comment out following lines to activate live plotting:
*import matplotlib*
*matplotlib.use('Agg')*

To Run:
----------
Can run main.py for individual control of inputs. Can run runBatch.py for batch executions.


