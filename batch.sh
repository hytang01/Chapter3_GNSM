#!/bin/bash
#SBATCH --job-name=test
#SBATCH --partition=serc
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1

module restore gprs-gcc10 

export ADGPRS_EXEC=/home/groups/lou/hytang/AD-GPRS/bin/ADGPRS/ADGPRS

#opt -- by default AD-GPRS take the max num of thread available 
#srun ${ADGPRS_EXEC} gprs.txt 

#opt.2.a -- to make sure we take 16 thread
${ADGPRS_EXEC} gprs.txt $SLURM_CPUS_PER_TASK

#opt.2.b -- to make sure we take 16 thread
#export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
#srun ${ADGPRS_EXEC} gprs.txt 


