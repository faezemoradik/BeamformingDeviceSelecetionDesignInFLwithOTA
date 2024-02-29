#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=17:00:00
#SBATCH --account=def-benliang
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
#SBATCH --mail-user=<faeze.moradi@mail.utoronto.ca>
#SBATCH --mail-type=ALL

source ~/projects/def-benliang/kalarde/ENV2/bin/activate
cd ~/projects/def-benliang/kalarde/BeamformingDeviceSelecetionDesignInFLwithOTA


python main.py -myseed 0 -dataset 'MNIST' -method 'Gibbs'