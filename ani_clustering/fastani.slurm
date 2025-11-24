#!/bin/bash
#SBATCH -p partition
#SBATCH -t day-hours
#SBATCH --mem=memoryG
#SBATCH -c cores
#SBATCH --job-name=fastani
#SBATCH --output=fastani.log
#SBATCH --mail-type=ALL
#SBATCH --mail-user=mail@address

#-------------Softwares--------------#
source "~/conda_init.sh"
conda activate fastani
#------------------------------------#

RUN_DIR='RUN_DIR'
mkdir -p "${RUN_DIR}"
cd "${RUN_DIR}" || exit

# Run fastANI
fastANI --ql input.txt --rl input.txt -o fastani.out --matrix -t "${SLURM_CPUS_PER_TASK:-1}"

echo 'JOB COMPLETED'
