#!/bin/bash
# Run it with sbatch file_name, check it with scontrol show job <job_id>
# ssh -p 6443 mikhaeldeo@host-ai.cs.ui.ac.id
# Then ssh -NfL localhost:<PORT>:10.119.105.199:<PORT> -p 6443 mikhaeldeo@host-ai.cs.ui.ac.id after getting the PORT and NODE_IP from scontrol show job <job_id>
# To access the jupyter notebook, open the browser and go to localhost:<PORT>
#SBATCH --job-name=singularity
#SBATCH --output=singularity-out-%u-%j.txt
#SBATCH --error=singularity-err-%u-%j.txt
#SBATCH --ntasks=1
#SBATCH --qos=1gpu
#SBATCH --partition=dgx1
#SBATCH --gpus=1
#SBATCH --mail-user=mikhael.deo@ui.ac.id	# Change <RECIPIENT_EMAIL> to your email, e.g. john.doe@mail.com
#SBATCH --mail-type=ALL

# Variables, please change the values according your needs
INSTANCE_NAME=python                        # The instance name
SIF_FILENAME=/srv/images/python_3.12.9.sif  # Image will be used
VENV_DIR=venv                               # Directory of virtual environment

# Creates a Singularity instance using provided image
export SINGULARITY_TMPDIR=$HOME/temp
singularity instance start --nv -f -w $SIF_FILENAME $INSTANCE_NAME

# Creates Python virtual environment with its directory named according to the value of VENV_DIR variable
cp /local-1/scripts/startup-script-v2.sh ./
singularity exec --cwd /root instance://$INSTANCE_NAME ./startup-script-v2.sh $VENV_DIR

# Starts Jupyter
singularity exec instance://$INSTANCE_NAME bash -c "
    source /root/$VENV_DIR/bin/activate;
    jupyter notebook --notebook-dir=/root --allow-root --ip=0.0.0.0;
    jupyter server list
"

# Keeps the job alive until it reaches the time limit
while true; do sleep 1; done

