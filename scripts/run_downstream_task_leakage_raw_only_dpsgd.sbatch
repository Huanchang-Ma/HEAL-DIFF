#!/bin/bash
#SBATCH --job-name=downstream-raw-dp
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=99g
#SBATCH --gres=gpu:1
#SBATCH --time=100:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

set -euo pipefail

module load gcc
module load cuda/12.1.0-gcc-8.5.0-5lkaqss
module load anaconda

CONDA_ENV="${CONDA_ENV:-ai}"
if command -v conda >/dev/null 2>&1; then
  export PS1="${PS1:-}"
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "${CONDA_ENV}"
else
  echo "conda not found in PATH"
  exit 1
fi

REPO_ROOT="/home/mhc/Diffusion_Model/Health_Care/"
MMHEALTH_ROOT="${REPO_ROOT}/MMHealth-main"

cd "${REPO_ROOT}"
mkdir -p outputs

echo "Host: $(hostname)"
echo "User: $(whoami)"
echo "Conda env: ${CONDA_ENV}"
echo "Python: $(command -v python)"

if ! python -c "import opacus" >/dev/null 2>&1; then
  python -m pip install --user opacus
  export PYTHONPATH="$(python -c 'import site; print(site.getusersitepackages())'):${PYTHONPATH:-}"
fi

python -m src.audit.downstream_task_leakage \
  --mmhealth-root "${MMHEALTH_ROOT}" \
  --task PHE \
  --dataset MIII \
  --model ours \
  --exp-num 20-raw-only-dpsgd-n1p0 \
  --feature-source raw_only \
  --dp-sgd \
  --dp-noise-multiplier 1.0 \
  --dp-max-grad-norm 1.0 \
  --dp-delta 1e-5 \
  --dp-physical-batch-size 8 \
  --agg mean \
  --fpr-targets 0.001,0.005,0.01,0.05 \
  --bootstrap-iters 1000 \
  --bootstrap-confidence 0.95 \
  --bootstrap-seed 528 \
  --subgroup-analysis \
  --min-subgroup-patients 30 \
  --train-if-missing \
  --output outputs/downstream_task_leakage_phe_miii_ours_raw_only_dpsgd_n1p0.json
