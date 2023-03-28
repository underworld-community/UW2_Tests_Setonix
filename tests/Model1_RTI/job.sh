#!/bin/bash -l
 
#SBATCH --account=pawsey0407
#SBATCH --job-name=uw2141-testcm
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=1  # OMP_NUM_THREADS equivalent
#SBATCH --time=01:00:00
#SBATCH --export=none
 

module load python/3.9.15 py-mpi4py/3.1.2-py3.9.15 py-numpy/1.20.3 py-h5py/3.4.0 py-cython/0.29.24
export OPT_DIR=/software/projects/pawsey0407/setonix/
source $OPT_DIR/py39/bin/activate
export PYTHONPATH=/software/projects/pawsey0407/setonix/underworld/2.14.2/lib/python3.9/site-packages/:$PYTHONPATH

export model="test_FreeSurface_Kaus2010_RTI.py"

# execute
srun -n $SLURM_NTASKS python3 $model
