Running the Code

After installation, this code may be run using the command

mpiexec -n N python test.py

where N is the desired number of cores. The variable of interest, i.e. pga, the radius, the desired number of realizations, and the path to the xml files may be set within the test.py file. 

Installation and Dependencies

This package depends on:

numpy, http://www.numpy.org/
matplotlib, http://matplotlib.org/index.html
scipy, http://www.scipy.org/scipylib/index.html

neicio, a Python package containing code modules extracted from the PAGER. May be found on GitHub
http://www.github.com/usgs/neicio

hazardlib, the openquake hazard library. May be found on GitHub
https://github.com/gem/oq-hazardlib

cartopy, used for plotting. Install using 
conda install -c ioos cartopy

mpi4py, MPI distribution for python from OpenMPI

This implementation was written using Anaconda
