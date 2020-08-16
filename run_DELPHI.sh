#!/bin/bash
#SBATCH --account=ctb-ilie_cpu
#SBATCH --partition=shadowfax
#SBATCH --nodelist=dus27
#SBATCH --cpus-per-task=32
#SBATCH --mem=1000G
#SBATCH --time=1-00:00

source /home/sbasak3/DELPHI/dusky/bin/activate

#set -x
# This is the DELPHI program entrance
# Usage: ./run_DELPHI.sh [INPUT_FN]
which python
export PRO_DIR=$PWD
export INPUT_FN=${PRO_DIR}/Datasets/Dset_72_Pid_Pseq.txt
export OUT_DIR=${PRO_DIR}/out-$(date +%Y-%m-%d-%H-%M-%S)
export TMP_DIR=${PRO_DIR}/tmp-$(date +%Y-%m-%d-%H-%M-%S)

echo "PRO_DIR: $PRO_DIR"
echo "TMP_DIR: $TMP_DIR"

####################
# compute features#
####################
bash feature_computation/compute_features.sh $INPUT_FN

# ####################
# #    run DELPHI    #
# ####################
python3 predict.py -i $INPUT_FN -d $TMP_DIR -o $OUT_DIR -c 1
if [ $? -ne 0 ]
then
   echo "[Error:] DELPHI returns 1!"
fi
