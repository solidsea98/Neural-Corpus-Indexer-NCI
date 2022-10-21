#!/usr/bin/env bash
export PYTHONPATH="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

python main.py --decode_embedding 2 --n_gpu 8 --mode train --query_type gtq_doc_aug_qg --adaptor_layer_num 4 \
--model_info base --train_batch_size 64 --eval_batch_size 64 --test1000 0 --dropout_rate 0.1 --Rdrop 0.15 \
--adaptor_decode 1 --adaptor_efficient 1 --aug_query 1 --aug_query_type corrupted_query --input_dropout 1 --id_class bert_k30_c30_1 \
--kary 30 --output_vocab_size 30 --doc_length 64 --denoising 0 --max_output_length 10 \
--trivia 0 --nq 1 #--resume_from_checkpoint '' 