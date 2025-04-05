#!/bin/bash
#SBATCH --job-name=setup_conda_env
#SBATCH --output=/home/cs-25-344/Cs-25-344-sdfad/src/outputs/%x_%j.out
#SBATCH --error=/home/cs-25-344/Cs-25-344-sdfad/src/outputs/%x_%j.err
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:0
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=anthonyde@vcu.edu

cd $SLURM_SUBMIT_DIR

# === CONFIGURATION ===
MINICONDA_DIR="/home/cs-25-344/miniconda3"
ENV_PATH="/home/cs-25-344/conda-envs/waymo-env"

# === 1. INSTALL MINICONDA IF MISSING ===
if [ ! -d "$MINICONDA_DIR" ]; then
    echo "Installing Miniconda to $MINICONDA_DIR..."
    curl -sSLO https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
    bash Miniconda3-latest-Linux-x86_64.sh -b -p "$MINICONDA_DIR"
    rm Miniconda3-latest-Linux-x86_64.sh
    chgrp -R 'egr cs-25-344' "$MINICONDA_DIR"
    chmod -R g+rwX,o-rwx "$MINICONDA_DIR"
fi

# Load conda into this shell
source "$MINICONDA_DIR/etc/profile.d/conda.sh"

# === Update conda to latest stable version ===
echo "Updating conda base install..."
conda update -n base -c defaults conda --yes

# === 2. CREATE ENV IF NEEDED ===
if [ ! -d "$ENV_PATH" ]; then
    echo "Creating conda env at $ENV_PATH..."
    conda create --yes --prefix "$ENV_PATH" python=3.10
    chgrp -R 'egr cs-25-344' "$ENV_PATH"
    chmod -R g+rwX,o-rwx "$ENV_PATH"
else
    echo "Conda env already exists at $ENV_PATH"
fi

# === 3. ACTIVATE ENV AND INSTALL PACKAGES ===
conda activate "$ENV_PATH"

echo "Installing conda packages..."
conda install --yes -c nvidia -c pytorch -c conda-forge \
    pytorch torchvision torchaudio pytorch-cuda=11.8 \
    numpy=1.24 matplotlib=3.5 seaborn statsmodels \
    pyarrow pillow openpyxl ipython tqdm scikit-learn


# === 4. INSTALL PIP PACKAGES ===
echo "Installing pip packages..."
pip install --no-cache-dir --no-deps waymo-open-dataset-tf-2-11-0 protobuf==3.20.*
pip install efficientnet_pytorch
pip install --force-reinstall --no-cache-dir pyarrow

# === 5. FIX PERMISSIONS AGAIN ===
chgrp -R 'egr cs-25-344' "$ENV_PATH"
chmod -R g+rwX,o-rwx "$ENV_PATH"

echo "✅ Conda setup complete at $(date)"