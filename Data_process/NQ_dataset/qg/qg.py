import pandas as pd
import pickle
import torch
import os
import re
import random
import csv
import jsonlines
import numpy as np
import pickle
import time
from tqdm import tqdm, trange
from sklearn.cluster import KMeans
from typing import Any, List, Sequence, Callable
from itertools import islice, zip_longest
from transformers import BertTokenizer, BertModel, AutoTokenizer, AutoModelForSeq2SeqLM
from sklearn.cluster import MiniBatchKMeans
import argparse

def pad_sequence_to_length(
        sequence: Sequence,
        desired_length: int,
        default_value: Callable[[], Any] = lambda: 0,
        padding_on_right: bool = True,
) -> List:
    sequence = list(sequence)
    # Truncates the sequence to the desired length.
    if padding_on_right:
        padded_sequence = sequence[:desired_length]
    else:
        padded_sequence = sequence[-desired_length:]
    # Continues to pad with default_value() until we reach the desired length.
    pad_length = desired_length - len(padded_sequence)
    # This just creates the default value once, so if it's a list, and if it gets mutated
    # later, it could cause subtle bugs. But the risk there is low, and this is much faster.
    values_to_pad = [default_value()] * pad_length
    if padding_on_right:
        padded_sequence = padded_sequence + values_to_pad
    else:
        padded_sequence = values_to_pad + padded_sequence
    return padded_sequence

def main(args):

    device=torch.device(f"cuda:{args.cuda_device}")
    ##  You can also download from Hugging Face. This folder should be in the same path as the notebook.
    model = AutoModelForSeq2SeqLM.from_pretrained("doc2query-t5-base-msmarco").to(f"cuda:{args.cuda_device}")


    # if torch.cuda.device_count() > 1:
    #     print("Let's use", torch.cuda.device_count(), "GPUs!")
    #     # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
    #     model=torch.nn.DataParallel(model, device_ids=[0,1]).cuda()
        
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained("doc2query-t5-base-msmarco")


    id_doc_dict = {}

    if 'NQ' in args.dataset:
        train_file = "../NQ_doc_content.tsv"
        with open(train_file, 'r') as f:
            for line in f.readlines():
                docid, _, _, content, _, _, _ = line.split("\t")
                id_doc_dict[docid] = content
    elif 'Trivia' in args.dataset:
        train_file = "../Trivia_doc_content.tsv"
        with open(train_file, 'r') as f:
            for line in f.readlines():
                _, docid, _, _, content = line.split("\t")
                id_doc_dict[docid] = content
    
    text_id_all = list(id_doc_dict.keys())
    text_list_all = [id_doc_dict[id_] for id_ in text_id_all]

    base = int(len(text_list_all) / args.partition_num)
    
    text_partitation = []
    text_partitation_id = []

    text_partitation.append(text_list_all[:base])
    text_partitation_id.append(text_id_all[:base])
    
    for i in range(args.partition_num-2):
        text_partitation.append(text_list_all[(i+1)*base: (i+2)*base])
        text_partitation_id.append(text_id_all[(i+1)*base: (i+2)*base])

    text_partitation.append(text_list_all[(i+2)*base:  ])
    text_partitation_id.append(text_id_all[(i+2)*base:  ])

    output_qg = []
    output_docid = []

    for i in trange(len(text_partitation[args.idx])):

        next_n_lines = text_partitation[args.idx][i]
        batch_input_ids = []
        sen = next_n_lines[:args.max_len]

        batch_input_ids.append(tokenizer.encode(text=sen, add_special_tokens=True))

        max_len = max([len(sen) for sen in batch_input_ids] )
        batch_input_ids = [
            pad_sequence_to_length(
                sequence=sen, desired_length=max_len, default_value=lambda : tokenizer.pad_token_id,
                padding_on_right=False
            ) for sen in batch_input_ids
        ]
        batch_input_ids = torch.tensor(data=batch_input_ids, dtype=torch.int64, device=device)

        generated = model.generate(
            input_ids=batch_input_ids,
            max_length=32,
            do_sample=True,
            num_return_sequences=args.return_num,
        )

        generated = tokenizer.batch_decode(sequences=generated.tolist(), skip_special_tokens=True)
        for index, g in enumerate(generated):
            output_qg.append(g)
            output_docid.append(text_partitation_id[args.idx][i])

    output = open(f'pkl/{args.dataset}_outpt_tensor_512_content_{args.max_len}_{args.return_num}_{args.idx}.pkl', 'wb', -1)
    pickle.dump(output_qg, output)
    output.close()

    output = open(f'pkl/{args.dataset}_outpt_tensor_512_content_{args.max_len}_{args.return_num}_{args.idx}_id.pkl', 'wb', -1)
    pickle.dump(output_docid, output)
    output.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Options for Commonsense Knowledge Base Completion')

    parser.add_argument("--idx", type=int, default=0, help="partitation")
    parser.add_argument("--partition_num", type=int, default=8, help="partitation")
    parser.add_argument("--dataset", type=str, default='NQ', help="partitation")
    parser.add_argument("--max_len", type=int, default=64, help="max length")
    parser.add_argument("--return_num", type=int, default=20, help="return num")
    parser.add_argument("--cuda_device", type=int, default=0, help="cuda")

    args = parser.parse_args()
    print(args)

    main(args)
