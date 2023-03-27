#!/bin/bash -l

module load python/3.9.15 py-mpi4py/3.1.2-py3.9.15 py-numpy/1.20.3 py-h5py/3.4.0 py-cython/0.29.24

export OPT_DIR=/software/projects/pawsey0407/setonix/
source $OPT_DIR/py39/bin/activate

# For development only
export PETSC_DIR=$OPT_DIR/petsc-3.18.1
export PYTHONPATH=$PETSC_DIR/:$PYTHONPATH

export PYTHONPATH=/software/projects/pawsey0407/setonix/underworld/2.14.2/lib/python3.9/site-packages/:$PYTHONPATH