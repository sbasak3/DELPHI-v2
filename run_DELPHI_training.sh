#!/bin/bash
#SBATCH --account=ctb-ilie_cpu
#SBATCH --partition=shadowfax
#SBATCH --nodelist=dus27
#SBATCH --cpus-per-task=32
#SBATCH --mem=1000G
#SBATCH --time=5-00:00

source /home/sbasak3/DELPHI/dusky/bin/activate

#set -x
# This is the DELPHI program entrance
# Usage: ./run_DELPHI_training.sh [INPUT_FN]
which python
export PRO_DIR=$PWD
export INPUT_FN=${PRO_DIR}/Datasets/training_Pid_Pseq.txt
export INPUT_FN_LABEL=${PRO_DIR}/Datasets/training_Pid_Pseq_label.txt
export TMP_DIR=${PRO_DIR}/tmp-$(date +%Y-%m-%d-%H-%M-%S)
echo "PRO_DIR: $PRO_DIR"
echo "TMP_DIR: $TMP_DIR"

####################
# compute features#
####################
bash feature_computation/compute_features.sh $INPUT_FN

# #############################
# #    run DELPHI training   #
# #############################
python3 train.py -i $INPUT_FN_LABEL -d $TMP_DIR -c 1 -ms 3 -ep 5
if [ $? -ne 0 ]
then
   echo "[Error:] DELPHI training returns error!"
fi