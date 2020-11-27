#!/bin/bash
#SBATCH --account=ctb-ilie_cpu
#SBATCH --partition=shadowfax
#SBATCH --nodelist=dus31
#SBATCH --cpus-per-task=12
#SBATCH --mem=250G
#SBATCH --time=5-00:00

source /home/sbasak3/DELPHI/dusky/bin/activate

#set -x
# This is the DELPHI program entrance
# Usage: ./run_DELPHI.sh [INPUT_FN]
which python
export PRO_DIR=$PWD
export INPUT_FN=${PRO_DIR}/Datasets/Dset_448_Pid_Pseq.txt
export INPUT_FN_LABEL=${PRO_DIR}/Datasets/Dset_448_Pid_Pseq_label.txt
export OUT_DIR=${PRO_DIR}/out-$(date +%Y-%m-%d-%H-%M-%S)
export TMP_DIR=${PRO_DIR}/tmp-$(date +%Y-%m-%d-%H-%M-%S)

echo "PRO_DIR: $PRO_DIR"
echo "TMP_DIR: $TMP_DIR"

#####################
#check PSSM database#
#####################
#PSSM_DIR=${TMP_DIR}/PSSM_raw/1/
#echo "load_PSSM_DB"
#python3 utils/load_PSSM_DB.py ${INPUT_FN} ${TMP_DIR}/PSSM_raw/1

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

# ################################
# #    Performance Evaluation   ##
# ################################

python3 ${PRO_DIR}/utils/performance_evaluation.py $OUT_DIR $INPUT_FN_LABEL Dset_448