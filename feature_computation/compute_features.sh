#!/bin/bash
#set -x
mkdir -p $TMP_DIR


# ####################
# ##   BERT   ##
# #################### (verified)
echo "computing BERT"
${PRO_DIR}/feature_computation/BERT/run_BERT.sh 

# ####################
# ##   ELMo   ##
# #################### (verified)
echo "computing ELMo"
${PRO_DIR}/feature_computation/ELMo/run_ELMo.sh 

# ####################
# ##   Albert   ##
# #################### (verified)
echo "computing Albert"
${PRO_DIR}/feature_computation/Albert/run_Albert.sh 

# ####################
# ##   XLNet   ##
# #################### (verified)
echo "computing XLNet"
${PRO_DIR}/feature_computation/XLNet/run_XLNet.sh 

# ####################
# ##   FastText   ##
# #################### (verified)
echo "computing FastText"
${PRO_DIR}/feature_computation/FastText/run_FastText.sh 

# ####################
# ##   Glove   ##
# #################### (verified)
echo "computing Glove"
${PRO_DIR}/feature_computation/Glove/run_Glove.sh 