WORKFLOW

1. Connect to Big Red 200 via PuTTY

Launch PuTTY.

Enter the hostname for Big Red 200 (bigred200.uits.iu.edu).

Enabling X11 forwarding if GUI applications are needed (Connection → SSH → X11 → Enable X11 forwarding).

Log in with your IU username and passphrase.

Ensure Xming is running on your local machine if X11 forwarding is enabled.

2. Transfer Python Code Using SCP

From the local machine, use scp to copy your Data and Python script to your home directory on Big Red 200:
Data:
mkdir -p /N/u/prpremk/BigRed200/brats_h5
scp -r "C:\Users\priya\Desktop\priyanka\Indiana_University\8_Deep_Learning\Project\archive\BraTS2020_training_data\content\data"   prpremk@bigred200.uits.iu.edu:/N/u/prpremk/BigRed200/brats_h5/

Script:
scp "C:/Users/priya/Desktop/priyanka/Indiana_University/8_Deep_Learning/Project/brats_multiresnet2.py" prpremk@bigred200.uits.iu.edu:/N/u/prpremk/BigRed200/

Check contents:
ls -l /N/u/prpremk/BigRed200/brats_h5/data

3. Prepare SLURM Job Script

open editor:
vi run_brats.sh

Create a SLURM script (e.g., run_job.sbatch) to submit your Python job:

#!/bin/bash
#SBATCH -J bratsjob          # Job name
#SBATCH -o bratsjob.out      # Standard output
#SBATCH -e brats.err          # Standard error
#SBATCH -A c01935            # IU project/account
#SBATCH -p gpu               # Partition
#SBATCH --gres=gpu:1         # Request 1 GPU
#SBATCH -N 1                 # Number of nodes
#SBATCH -n 4                 # Number of CPU tasks
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00      # Max runtime (hh:mm:ss)

module load python/gpu/3.10.10

export TF_FORCE_GPU_ALLOW_GROWTH=true

python -u ~/brats_multiresnet2.py


4. Submit the Job

From the terminal on Big Red 200:
sbatch run_brats.sh

5. Monitor Job Status

Check job status using:

squeue -u prpremk

After the job completes, check output and error logs:
cat bratsjob.out
cat  brats.err


6. Retrieve Results

Use scp to copy output files or generated figures back to your local machine:

scp prpremk@bigred200.uits.iu.edu:/N/u/prpremk/BigRed200/brats_results/test_slice0_WT.png C:\Users\priya\Desktop\priyanka\Indiana_University\8_Deep_Learning\Project\brats_results\
