#!/bin/bash
compute="python3 ${PRO_DIR}/feature_computation/FastText/compute.py"
${compute} ${INPUT_FN} ${TMP_DIR}/FastText.txt