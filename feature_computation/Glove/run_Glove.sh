#!/bin/bash
compute="python3 ${PRO_DIR}/feature_computation/Glove/compute.py"
${compute} ${INPUT_FN} ${TMP_DIR}/Glove.txt