#!/bin/bash -l
 
#SBATCH --account=pawsey0407
#SBATCH --job-name=uw2141-testcm
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=1  # OMP_NUM_THREADS equivalent
#SBATCH --time=01:00:00
#SBATCH --export=none
 
# load Singularity
module load singularity/3.8.6
 
# Define the container to use
export pawseyRepository=/scratch/pawsey0407/jgiordani/myimages/

export containerImage=$pawseyRepository/underworld2-mpich_latest.sif
export model="test_FreeSurface_Kaus2010_RTI.py"

# ## uncomment for the setonix pure h5py test ##
# export containerImage=$pawseyRepository/hpc-python_2022.03-hdf5mpi.sif
# export model="parallel_write.py"

# as per
# https://support.pawsey.org.au/documentation/pages/viewpage.action?pageId=116131367#UsewithSingularity-RunningPythonandR
# we unset all the host python-related ENV vars
unset $( env | grep ^PYTHON | cut -d = -f 1 | xargs )

# execute
srun --export=all -u -n $SLURM_NTASKS singularity exec -B ${PWD}:/work $containerImage bash -c "cd /work/; python3 $model"
