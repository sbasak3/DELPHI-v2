#!/bin/bash
mkdir -p ${TMP_DIR}/BERT_raw/

data_prep="python3 ${PRO_DIR}/feature_computation/BERT/data_prep_kmers.py"
${data_prep} ${INPUT_FN} ${TMP_DIR}/BERT_raw/BERT_Input.txt

python ${PRO_DIR}/feature_computation/BERT/extract_features.py \
  --input_file=${TMP_DIR}/BERT_raw/BERT_Input.txt \
  --output_file=${TMP_DIR}/BERT_raw/BERT_Output.txt \
  --vocab_file=${PRO_DIR}/feature_computation/BERT/vocab_3mer.txt \
  --do_lower_case=False \
  --bert_config_file=${PRO_DIR}/feature_computation/BERT/bert_config_3mer.json \
  --init_checkpoint=${PRO_DIR}/feature_computation/BERT/pretraining_output_3mer/model.ckpt-20 \
  --layers=-1 \
  --max_seq_length=2048 \
  --batch_size=1

compute="python3 ${PRO_DIR}/feature_computation/BERT/compute.py"
${compute} ${INPUT_FN} ${TMP_DIR}/BERT_raw/BERT_Output.txt ${TMP_DIR}/BERT_3mer.txt
