#!/bin/bash
#SBATCH -p partition
#SBATCH -t day-hours
#SBATCH --mem=memoryG
#SBATCH -c cores
#SBATCH --job-name=cluster_ani
#SBATCH --output=cluster_ani.log
#SBATCH --mail-type=ALL
#SBATCH --mail-user=mail@address

#-------------Softwares--------------#
source "~/conda_init.sh"
conda activate notebook
#------------------------------------#

RUN_DIR='RUN_DIR'
mkdir -p "${RUN_DIR}"
cd "${RUN_DIR}" || exit

# Run fastANI
cluster_ani.py \
	--ani-matrix fastani.matrix \
	--input-list input_list.tsv \
	--metadata metadata.csv \
	--threshold 0.95 \
	--outdir "${RUN_DIR}" \
	--threads "${SLURM_CPUS_PER_TASK:-1}" \
	--debug

echo 'JOB COMPLETED'