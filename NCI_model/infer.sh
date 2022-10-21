#!/usr/bin/env bash

# CKPT path
INFER_CKPT='/sunhao_data/sunhao/repo/corpus_code_yingyan/logs/nq:_1_trivia:_0_kary:_30_gtq_doc_aug_qg20_large_bert_k30_c30_2_univ_NQ_dev_recall_dem:_2_ada:_1_adaeff:_1_adanum:_2_RDrop:_0.1_0.1_0__lre2.0d1.0_epoch=23-recall1=0.657048.ckpt' 
BEAM_SIZE=100

# Set args to be the same as training.
# CUDA_VISIBLE_DEVICES=1 
python main.py --decode_embedding 2 --n_gpu 1 --mode eval --query_type gtq_doc_aug_qg --adaptor_layer_num 2 \
--infer_ckpt $INFER_CKPT --num_return_sequences $BEAM_SIZE --tree 1 \
--model_info large --train_batch_size 64 --eval_batch_size 4 --test1000 0 --dropout_rate 0.1 --Rdrop 0.1 \
--adaptor_decode 1 --adaptor_efficient 1 --aug_query 1 --aug_query_type corrupted_query --input_dropout 1 --id_class bert_k30_c30_5 \
--kary 30 --output_vocab_size 30 --doc_length 64 --denoising 0 --max_output_length 10 \
--trivia 0 --nq 1 
