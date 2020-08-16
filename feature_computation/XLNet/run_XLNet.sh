#!/bin/bash
compute="python3 ${PRO_DIR}/feature_computation/XLNet/compute.py"
${compute} ${INPUT_FN} ${TMP_DIR}/XLNet.txt