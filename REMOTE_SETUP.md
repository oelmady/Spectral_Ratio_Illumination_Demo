# Remote Server Setup Guide: Northeastern Explorer HPC

Step-by-step instructions to deploy and run your project on Northeastern's Explorer HPC cluster.

## Phase 1: Initial Connection & Project Setup

### Step 1.1: SSH into Explorer Cluster
```bash
ssh your_northeastern_username@login.explorer.northeastern.edu
```

**What to expect:**
- You'll be prompted for your Northeastern password
- After entering it, you should see the Explorer welcome banner with the Explorer ASCII art
- Terminal prompt should look like: `[your_username@explorer-XX ~]$`

### Step 1.2: Verify Python and Load Modules
```bash
# First, see what modules are available on the cluster
module avail python

# Load a Python module (typically 3.10 or higher)
module load python/3.11  # or whichever version is available

# Verify it worked
python3 --version
```

**What to expect:**
- `module avail python` shows available Python versions
- Load the newest Python version you see listed
- `python3 --version` should return Python 3.10+ (not 2.7)

### Step 1.2b (Optional): Set Up Passwordless SSH
This is convenient so you don't have to type your password every time. **Perform these steps once.**

**On your LOCAL Mac (not connected to cluster):**
```bash
# Generate SSH keys (if you don't have them already)
cd ~/.ssh
ssh-keygen -t rsa
# Press Enter on all prompts, don't set a passphrase

# Copy keys to Explorer cluster
ssh-copy-id -i ~/.ssh/id_rsa.pub your_northeastern_username@login.explorer.northeastern.edu
# Enter your NU password when prompted

# Test passwordless login
ssh your_northeastern_username@login.explorer.northeastern.edu
# Should connect WITHOUT asking for password
```

Now future logins won't require a password.

### Step 1.3: Clone Your Project Repository
```bash
git clone https://github.com/oelmady/Spectral_Ratio_Illumination_Demo.git
cd Spectral_Ratio_Illumination_Demo
```

**What this does:**
- Downloads all your code files (except the 528MB model, which git-lfs will handle next)
- Creates a `Spectral_Ratio_Illumination_Demo` directory
- Navigates into the project folder

### Step 1.4: Download the Model Using Git-LFS
```bash
# Check if git-lfs is already installed
git lfs version

# If installed, pull the model (this may take 1-2 minutes)
git lfs pull --include="model/UNET_run_x10_01_last_model.pth"

# Verify the model was downloaded (should be ~528MB, not 134 bytes)
ls -lh model/UNET_run_x10_01_last_model.pth
```

**Expected output for model file:**
```
-rw-r--r-- 1 username group 528M Dec 4 12:34 model/UNET_run_x10_01_last_model.pth
```

If git-lfs is NOT installed on the server, ask your professor to install it, or they may have already pulled it for you.

---

## Phase 2: Environment Setup

### Step 2.1: Load Required Modules on Explorer
HPC clusters use "modules" to manage software environments. Do this every time you log in.

```bash
# Load Python (3.11 or latest available)
module load python/3.11

# Load libraries needed for PyTorch and OpenCV
module load gcc  # C++ compiler
module load cuda  # Optional: if you want GPU support

# Verify modules loaded
module list
```

**What to expect:**
- `module list` shows `python/3.11`, `gcc`, and optionally `cuda`

### Step 2.2: Create Python Virtual Environment
```bash
# Navigate to your project
cd ~/Spectral_Ratio_Illumination_Demo

# Create a virtual environment
python3 -m venv venv

# Activate it
source venv/bin/activate
# You should see (venv) in your terminal prompt

# Upgrade pip
pip install --upgrade pip
```

**Note:** You'll need to do `source venv/bin/activate` and load modules every time you connect to the cluster.

### Step 2.3: Install Project Dependencies
```bash
# Make sure you're in the project directory with venv activated
cd ~/Spectral_Ratio_Illumination_Demo
source venv/bin/activate

# Install all required packages
pip install -r requirements.txt

# This will install: opencv-python, torch, torchvision, numpy, matplotlib
```

**What to expect:**
- Installation may take 3-10 minutes (PyTorch is large)
- You should see `Successfully installed` messages for each package
- On HPC, pip may need to compile some packages; this is normal

### Step 2.4: Verify Installation
```bash
# Test that all imports work
python preflight_check.py
```

**Expected output:**
```
========================================
PREFLIGHT CHECK
========================================
✓ All imports successful
✓ Project structure valid
✓ Algorithm implementations present
✓ Model checkpoint found (528MB)
✓ Data directory exists

STATUS: READY FOR EXPERIMENTS
========================================
```

If any checks fail, see troubleshooting section at the end.

---

## Phase 3: Running Experiments

### Step 3.1: Prepare Environment for Each Session
Every time you connect to Explorer, run these commands:

```bash
# SSH to cluster
ssh your_northeastern_username@login.explorer.northeastern.edu

# Load modules
module load python/3.11 gcc

# Navigate to project
cd ~/Spectral_Ratio_Illumination_Demo

# Activate virtual environment
source venv/bin/activate
```

### Step 3.2: Prepare Input Data
Your input images should be 16-bit linear RGB TIFF files in `data/images/`:

```bash
# Check what data is already there
ls -la data/images/

# If you need to upload images from your local machine:
# On your LOCAL Mac (in a separate terminal):
# scp /path/to/your/image.tif your_northeastern_username@login.explorer.northeastern.edu:~/Spectral_Ratio_Illumination_Demo/data/images/
```

### Step 3.3: Run the Batch Processor (Interactive)
For testing and small runs (5-10 images), run interactively:

```bash
# Make sure venv is activated and modules loaded
source venv/bin/activate

# Run the batch processor
python scripts/run_batch.py \
  --use-model \
  --retinex \
  --baseline-retinex \
  --sr-correct \
  --iterations 5 \
  --sigma 15 \
  --distance 1.0
```

**Expected runtime:**
- Model inference: ~30 seconds per image (CPU on login node)
- Retinex algorithms: ~5-10 seconds per image
- Total for 5 images: ~3-5 minutes

### Step 3.4: Run Large Experiments (Optional: Batch Job Submission)
For processing many images or parameter sweeps, submit a job to the scheduler instead of running on login nodes.

**Create a file `run_experiment.sh`:**
```bash
#!/bin/bash
#SBATCH --job-name=spectral_retinex
#SBATCH --time=02:00:00          # 2 hour time limit
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=4        # Use 4 CPU cores
#SBATCH --mem=16gb               # Request 16GB memory

# Load modules
module load python/3.11 gcc

# Activate environment
source venv/bin/activate

# Run experiments
python scripts/run_batch.py \
  --use-model \
  --retinex \
  --baseline-retinex \
  --sr-correct \
  --iterations 5 \
  --sigma 15
```

Submit the job:
```bash
sbatch run_experiment.sh
```

Check job status:
```bash
squeue -u your_username
```

This allows long-running experiments without blocking the login node.

---

## Phase 4: Troubleshooting

### Issue: `module load python/3.11` says module not found
**Cause:** Different Python versions available on your cluster
**Solution:**
```bash
# Check available versions
module avail python

# Load a different version (e.g., 3.10, 3.12)
module load python/3.10
```

### Issue: `git lfs pull` returns empty or small file (after clone)
**Cause:** git-lfs may not be installed on cluster
**Solution:**
```bash
# Check if it's installed
git lfs version

# If not, ask RC to install it, or email rchelp@northeastern.edu
# Note: git lfs should already be installed on Explorer
```

### Issue: `ModuleNotFoundError: No module named 'torch'`
**Cause:** Dependencies not installed or venv not activated
**Solution:**
```bash
# Make sure modules are loaded
module load python/3.11 gcc

# Make sure venv is activated (you should see (venv) in prompt)
source venv/bin/activate

# Reinstall
pip install -r requirements.txt
```

### Issue: `pip install` is very slow or times out
**Cause:** Network bandwidth on login nodes is limited
**Solution:** Use a batch job (SBATCH) instead of running on login nodes. See Step 3.4.

### Issue: `PermissionError: [Errno 13] Permission denied: 'results/'`
**Cause:** Directory doesn't exist or wrong permissions
**Solution:**
```bash
mkdir -p results/
chmod 755 results/
```

### Issue: Model inference is extremely slow (>2 minutes per image)
**Cause:** Running on shared login node resources
**Solution:** Submit as batch job (Step 3.4) to get dedicated CPU cores

---

## Quick Reference: Session Checklist

Every time you connect to Explorer:

```bash
# 1. SSH to cluster
ssh your_northeastern_username@login.explorer.northeastern.edu

# 2. Load modules
module load python/3.11 gcc

# 3. Navigate to project
cd ~/Spectral_Ratio_Illumination_Demo

# 4. Activate venv
source venv/bin/activate

# 5. Run experiments
python scripts/run_batch.py --use-model --retinex --baseline-retinex --sr-correct

# 6. Download results (from LOCAL Mac terminal)
# scp your_northeastern_username@login.explorer.northeastern.edu:~/Spectral_Ratio_Illumination_Demo/results.tar.gz ./
```

### Useful HPC Commands

```bash
# See available modules
module avail

# See currently loaded modules
module list

# Unload a module
module unload python/3.11

# Submit a job
sbatch run_experiment.sh

# Check job status
squeue -u your_username

# Cancel a job
scancel job_id

# See past jobs
sacct -u your_username
```

---

## Important: Storage Locations on Explorer

- **Home directory:** `~/` (10GB quota, backed up daily)
- **Scratch space:** `/scratch` (temporary, no backup)
- **Project space:** `/project` (if assigned, larger quota)

Put your code in home (`~/Spectral_Ratio_Illumination_Demo`), but move large `results/` to scratch if storage becomes an issue.

---

## Contact If Issues:

- **Python/module issues:** Email rchelp@northeastern.edu
- **Git-lfs issues:** Email rchelp@northeastern.edu
- **Code/algorithm issues:** Ask your professor

---

## Success Indicators
✓ `preflight_check.py` returns all green checks
✓ Model file is ~528MB (not 134 bytes)
✓ `python scripts/run_batch.py --use-model` completes without errors on login node
✓ `results/` directory contains TIFF and PNG files
✓ You can download results back to your Mac with `scp`
