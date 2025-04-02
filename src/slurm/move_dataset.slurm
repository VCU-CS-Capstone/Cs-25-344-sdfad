#!/bin/bash
#SBATCH --job-name=waymo-transfer
#SBATCH --output=transfer.log
#SBATCH --error=transfer.err
#SBATCH --time=12:00:00             # Adjust time based on expected size
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --partition=standard        # or whatever partition your group uses

# Optional: Set group access for files
newgrp 'egr cs-25-344'
umask 007

# Load modules if needed (not likely for rsync)
# module load rsync

# Begin transfer
rsync -avz --progress anthonyde@tesla.cs.vcu.edu:/path/to/partial/ /home/cs-25-344/waymo_data/
