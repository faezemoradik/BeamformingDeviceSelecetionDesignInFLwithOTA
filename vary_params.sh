#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=00:10:00
#SBATCH --account=def-benliang
#SBATCH --mail-user=faeze.moradi@mail.utoronto.ca
#SBATCH --mail-type=ALL


for me in 'SelectAll' 'TopOne' 'Gibbs' 'GSDS' 'ADSBF' 'DC'; do
  sbatch --export=ME=${me} job.sh
done
 
