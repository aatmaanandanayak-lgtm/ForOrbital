#!/bin/bash
#SBATCH --job-name=chla_frontier_orb
#SBATCH --account=u_costaa
#SBATCH --partition=ncpu
#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --cpus-per-task=1
#SBATCH --mem=48G
#SBATCH --time=48:00:00
#SBATCH --output=../logs/%x_%j.out
#SBATCH --error=../logs/%x_%j.err

module purge
source ~/miniforge3/etc/profile.d/conda.sh
conda deactivate 2>/dev/null || true
module load ORCA/5.0.4-gompi-2022a
ORCA_EXE=$(which orca)

cd ~/working/Aatmaananda/others/GNN_spectpredict/orca_inputs/phase3_test
$ORCA_EXE chla_frontier_orbitals.inp > chla_frontier_orbitals.out 2>&1

echo "Chl a frontier orbital calculation finished."
