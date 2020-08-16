#!/bin/bash
compute="python3 ${PRO_DIR}/feature_computation/ELMo/compute.py"
${compute} ${INPUT_FN} ${TMP_DIR}/ELMo_n.txt