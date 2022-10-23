import os
import pickle
import random
from time import time
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler

from main_helper_loss import loss_zoo
from main_utils import assert_all_frozen, load_data_infer, \
    load_data, numerical_decoder, dec_2d
from transformers import (
    AdamW,
    T5ForConditionalGeneration,
    T5Tokenizer,
    T5Config,
    get_linear_schedule_with_warmup
)
from tqdm import tqdm



class Node(object):
    def __init__(self, token_id) -> None:
        self.token_id = token_id
        self.children = {}

    def __str__(self, level=0):
        ret = "\t" * level + repr(self.token_id) + "\n"
        for child in self.children.values():
            ret += child.__str__(level + 1)
        return ret

    def __repr__(self):
        return '<tree node representation>'


class TreeBuilder(object):
    def __init__(self) -> None:
        self.root = Node(0)

    def build(self) -> Node:
        return self.root

    def add(self, seq) -> None:
        '''
        seq is List[Int] representing id, without leading pad_token(hardcoded 0), with trailing eos_token(hardcoded 1) and (possible) pad_token, every int is token index
        e.g: [ 9, 14, 27, 38, 47, 58, 62,  1,  0,  0,  0,  0,  0,  0,  0]
        '''
        cur = self.root
        for tok in seq:
            if tok == 0:  # reach pad_token
                return
            if tok not in cur.children:
                cur.children[tok] = Node(tok)
            cur = cur.children[tok]


def encode_single_newid(args, seq):
    '''
    Param:
        seq: doc_id string to be encoded, like "23456"
    Return:
        List[Int]: encoded tokens
    '''
    target_id_int = []
    if args.kary:
        for i, c in enumerate(seq.split('-')):
            if args.position:
                cur_token = i * args.kary + int(c) + 2
            else:
                cur_token = int(c) + 2
            target_id_int.append(cur_token)
    else:
        for i, c in enumerate(seq):
            if args.position:
                cur_token = i * 10 + int(c) + 2  # hardcoded vocab_size = 10
            else:
                cur_token = int(c) + 2
            target_id_int.append(cur_token)
    return target_id_int + [1]  # append eos_token


def decode_token(args, seqs):
    '''
    Param:
        seqs: 2d ndarray to be decoded
    Return:
        doc_id string, List[str]
    '''
    result = []
    for seq in seqs:
        try:
            eos_idx = seq.tolist().index(1)
            seq = seq[1: eos_idx]
        except:
            print("no eos token found")
        if args.position:
            offset = np.arange(len(seq)) * args.output_vocab_size + 2
        else:
            offset = 2
        res = seq - offset
        #assert np.all(res >= 0)
        if args.kary:
            result.append('-'.join(str(c) for c in res))
        else:
            result.append(''.join(str(c) for c in res))
    return result

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

class l1_query(Dataset):
    def __init__(self, args, tokenizer, num_samples, print_text=False, task='train'):
        assert task in ['train', 'test']
        self.args = args
        input_length = args.max_input_length
        output_length = args.max_output_length * int(np.log10(args.output_vocab_size))
        inf_input_length = args.inf_max_input_length
        random_gen = args.random_gen
        softmax = args.softmax
        aug = args.aug

        if task == 'train':
            self.dataset, self.q_emb, self.query_dict, \
            self.prefix_embedding, self.prefix_mask, self.prefix2idx_dict = \
                load_data(args)
        elif task == 'test':
            self.dataset = load_data_infer(args)
            self.q_emb, self.query_dict, \
            self.prefix_embedding, self.prefix_mask, self.prefix2idx_dict \
                = None, None, None, None, None
        else:
            raise NotImplementedError("No Corresponding Task.")

        if num_samples:
            self.dataset = self.dataset[:num_samples]

        self.task = task
        self.input_length = input_length
        self.doc_length = self.args.doc_length
        self.inf_input_length = inf_input_length
        self.tokenizer = tokenizer
        self.output_length = output_length
        self.print_text = print_text
        self.softmax = softmax
        self.aug = aug
        self.random_gen = random_gen
        if random_gen:
            assert len(self.dataset[0]) >= 3
        self.random_min = 2
        self.random_max = 6
        self.vocabs = set(self.tokenizer.get_vocab().keys())
        for token in [self.tokenizer.eos_token, self.tokenizer.unk_token, self.tokenizer.sep_token,
                      self.tokenizer.pad_token, self.tokenizer.cls_token,
                      self.tokenizer.mask_token] + tokenizer.additional_special_tokens:
            if token is not None:
                self.vocabs.remove(token)

    def __len__(self):
        return len(self.dataset)

    @staticmethod
    def clean_text(text):
        text = text.replace('\n', '')
        text = text.replace('``', '')
        text = text.replace('"', '')

        return text


    def convert_to_features(self, example_batch, length_constraint):
        # Tokenize contexts and questions (as pairs of inputs)

        input_ = self.clean_text(example_batch)
        output_ = self.tokenizer.batch_encode_plus([input_], max_length=length_constraint,
                                                  padding='max_length', truncation=True, return_tensors="pt")

        return output_


    def __getitem__(self, index):
        inputs = self.dataset[index]
        query_embedding = torch.tensor([0])
        prefix_embedding, prefix_mask = torch.tensor([0]), torch.tensor([0])
        # if self.args.old_data:
        if len(inputs) >= 6:
            query, target, rank, neg_target, aug_query = inputs[0], inputs[1], inputs[2], inputs[4], inputs[5]
        elif len(inputs) >= 5:
            query, target, rank, neg_target = inputs[0], inputs[1], inputs[2], inputs[4]
        else:
            query, target, rank = inputs[0], inputs[1], inputs[2]

        if hasattr(self, 'query_dict') and self.query_dict is not None:
            query_embedding = self.q_emb[self.query_dict[query]]
        neg_targets_list = []
        if self.args.hard_negative:
            neg_targets_list = np.random.choice(neg_target, self.args.sample_neg_num)
        if self.args.aug_query and len(aug_query) >= 1:
            aug_query = np.random.choice(aug_query, 1)[0]
        else:
            aug_query = ""
        if self.args.label_length_cutoff:
            target = target[:self.args.max_output_length - 2]

        source = self.convert_to_features(query, self.input_length if self.task=='train' else self.inf_input_length)
        source_ids = source["input_ids"].squeeze()
        if 'print_token' in self.args.query_type:
            print("Input Text: ", query, '\n', "Output Text: ", source_ids)
        src_mask = source["attention_mask"].squeeze()
        aug_source = self.convert_to_features(aug_query, self.input_length if self.task=='train' else self.inf_input_length)
        aug_source_ids = aug_source["input_ids"].squeeze()
        aug_source_mask = aug_source["attention_mask"].squeeze()
        if self.args.multiple_decoder:
            target_ids, target_mask = [], []
            for i in range(self.args.decoder_num):
                targets = self.convert_to_features(target[i], self.output_length)
                target_ids.append(targets["input_ids"].squeeze())
                target_mask.append(targets["attention_mask"].squeeze())
        else:
            targets = self.convert_to_features(target, self.output_length)
            target_ids = targets["input_ids"].squeeze()
            target_mask = targets["attention_mask"].squeeze()

        def target_to_prefix_emb(target, tgt_length):
            tgt_prefix_emb = []
            prefix_masks = []
            for i in range(tgt_length):
                if i < len(target):
                    ###### fake data
                    _prefix_emb = np.random.rand(10, 768)
                    ###### real data
                    # _prefix_emb = self.prefix_embedding[self.prefix2idx_dict[target[:i]]]
                    _prefix_emb = torch.tensor(_prefix_emb)
                    tgt_prefix_emb.append(_prefix_emb.unsqueeze(0))
                    ##############################
                    ###### fake data
                    _prefix_mask = np.random.rand(10,)
                    _prefix_mask[_prefix_mask < 0.5] = 0
                    _prefix_mask[_prefix_mask > 0.5] = 1
                    ###### real data
                    # _prefix_mask = self.prefix_mask[self.prefix2idx_dict[target[:i]]]
                    _prefix_mask = torch.LongTensor(_prefix_mask)
                    prefix_masks.append(_prefix_mask.unsqueeze(0))
                    ##############################
                else:
                    tgt_prefix_emb.append(torch.zeros((1, 10,768)))
                    prefix_masks.append(torch.zeros((1, 10)))
            return torch.cat(tgt_prefix_emb, dim=0), torch.cat(prefix_masks, dim=0)

        if self.prefix_embedding is not None:
            prefix_embedding, prefix_mask = target_to_prefix_emb(target, self.output_length)

        neg_target_ids_list = []
        neg_target_mask_list = []
        neg_rank_list = []

        if self.args.hard_negative:
            for cur_target in neg_targets_list:
                cur_targets = self.convert_to_features(cur_target, self.output_length)
                cur_target_ids = cur_targets["input_ids"].squeeze()
                cur_target_mask = cur_targets["attention_mask"].squeeze()
                neg_target_ids_list.append(cur_target_ids)
                neg_target_mask_list.append(cur_target_mask)
                neg_rank_list.append(999)     #denote hard nagative

        lm_labels = torch.zeros(self.args.max_output_length, dtype=torch.long)

        if self.args.decode_embedding:
            ## func target_id+target_id2, twice or k
            def decode_embedding_process(target_ids):
                target_id = self.tokenizer.decode(target_ids)
                target_id_int = []
                if self.args.kary:
                    idx = 0 
                    target_id = target_id.split('-')
                    for i in range(0, len(target_id)):
                        c = target_id[i]
                        if self.args.position:
                            temp = i * self.args.output_vocab_size + int(c) + 2 \
                                if not self.args.hierarchic_decode else int(c) + 2
                        else:
                            temp = int(c) + 2
                        target_id_int.append(temp)
                else:
                    bits = int(np.log10(self.args.output_vocab_size))
                    idx = 0
                    for i in range(0, len(target_id), bits):
                        if i + bits >= len(target_id):
                            c = target_id[i:]
                        c = target_id[i:i + bits]
                        if self.args.position:
                            temp = idx * self.args.output_vocab_size + int(c) + 2 \
                                if not self.args.hierarchic_decode else int(c) + 2
                        else:
                            temp = int(c) + 2
                        target_id_int.append(temp)
                        idx += 1
                lm_labels[:len(target_id_int)] = torch.LongTensor(target_id_int)
                lm_labels[len(target_id_int)] = 1
                decoder_attention_mask = lm_labels.clone()
                decoder_attention_mask[decoder_attention_mask != 0] = 1
                target_ids = lm_labels
                target_mask = decoder_attention_mask
                return target_ids, target_mask

            if self.args.multiple_decoder:
                target_mask = []
                for i in range(len(target_ids)):
                    target_ids[i], cur_target_mask = decode_embedding_process(target_ids[i])
                    target_mask.append(cur_target_mask)
            else:
                target_ids, target_mask = decode_embedding_process(target_ids)

            if self.args.hard_negative:
                for i in range(len(neg_target_ids_list)):
                    cur_target_ids = neg_target_ids_list[i]
                    cur_target_ids, cur_target_mask = decode_embedding_process(cur_target_ids)
                    neg_target_ids_list[i] = cur_target_ids
                    neg_target_mask_list[i] = cur_target_mask

        return {"source_ids": source_ids,
                "source_mask": src_mask,
                "aug_source_ids": aug_source_ids,
                "aug_source_mask": aug_source_mask,
                "target_ids": target_ids,
                "target_mask": target_mask,
                "neg_target_ids": neg_target_ids_list,
                "neg_rank": neg_rank_list,
                "neg_target_mask": neg_target_mask_list,
                "doc_ids": doc_ids if self.args.contrastive_variant != '' else torch.tensor([-1997], dtype=torch.int64),
                "doc_mask": doc_mask if self.args.contrastive_variant != '' else torch.tensor([-1997], dtype=torch.int64),
                "softmax_index": torch.tensor([inputs[-1]], dtype=torch.int64)
                                        if self.softmax else torch.tensor([-1997], dtype=torch.int64),
                "rank": rank,
                "query_emb":query_embedding,
                "prefix_emb":prefix_embedding,
                "prefix_mask":prefix_mask}


class T5FineTuner(pl.LightningModule):
    def __init__(self, args, train=True):
        super(T5FineTuner, self).__init__()

        tree_save_path = args.output_dir + args.query_info + 'tree.pkl'
        if os.path.isfile(tree_save_path):
            print('tree not true')
            with open(tree_save_path, "rb") as input_file:
                root = pickle.load(input_file)
            self.root = root
        else:
            print("Begin build tree")
            builder = TreeBuilder()
            if args.trivia:
                train_file = '../Data_process/trivia_dataset/train.tsv'
                dev_file = '../Data_process/trivia_dataset/dev.tsv'
                test_file = '../Data_process/trivia_dataset/test.tsv'
                df_train = pd.read_csv(
                    train_file,
                    encoding='utf-8', names=["query", "queryid", "oldid", "bert_k30_c30_1", "bert_k30_c30_2", "bert_k30_c30_3", "bert_k30_c30_4", "bert_k30_c30_5"],
                    header=None, sep='\t', dtype={'query': str, 'queryid': str, 'oldid': str, args.id_class: str}).loc[:, ["query", args.id_class]]
                df_dev = pd.read_csv(
                    dev_file,
                   encoding='utf-8', names=["query", "queryid", "oldid", "bert_k30_c30_1", "bert_k30_c30_2", "bert_k30_c30_3", "bert_k30_c30_4", "bert_k30_c30_5"],
                    header=None, sep='\t', dtype={'query': str, 'queryid': str, 'oldid': str, args.id_class: str}).loc[:, ["query", args.id_class]]
                df_test = pd.read_csv(
                    test_file,
                    encoding='utf-8', names=["query", "queryid", "oldid", "bert_k30_c30_1", "bert_k30_c30_2", "bert_k30_c30_3", "bert_k30_c30_4", "bert_k30_c30_5"],
                    header=None, sep='\t', dtype={'query': str, 'queryid': str, 'oldid': str, args.id_class: str}).loc[:, ["query", args.id_class]]
                df_dev = pd.merge(df_test, df_dev, how='outer')
                df = pd.merge(df_train, df_dev, how='outer')
            elif args.nq:
                train_filename = '../Data_process/NQ_dataset/nq_train_doc_newid.tsv'
                dev_filename = '../Data_process/NQ_dataset/nq_dev_doc_newid.tsv'
                df_train = pd.read_csv(
                    train_filename,
                    names=["query", "queryid", "oldid", "bert_k30_c30_1", "bert_k30_c30_2", "bert_k30_c30_3", "bert_k30_c30_4", "bert_k30_c30_5"],
                    encoding='utf-8', header=None, sep='\t',
                    dtype={'query': str, 'queryid': str, 'oldid': str, args.id_class: str}).loc[:,
                    ["query", args.id_class]]
                df_dev = pd.read_csv(
                    dev_filename,
                    names=["query", "queryid", "oldid", "bert_k30_c30_1", "bert_k30_c30_2", "bert_k30_c30_3", "bert_k30_c30_4", "bert_k30_c30_5"],
                    encoding='utf-8', header=None, sep='\t',
                    dtype={'query': str, 'queryid': str, 'oldid': str, args.id_class: str}).loc[:,
                        ["query", args.id_class]]
                df = pd.merge(df_train, df_dev, how='outer')

            for _, (_, newid) in tqdm(df.iterrows()):
                if args.label_length_cutoff:
                    newid = newid[:args.max_output_length-2]
                if args.trivia:
                    newid = newid.split(",")
                    for i in range(len(newid)):
                        toks = encode_single_newid(args, newid[i])
                        builder.add(toks)
                elif args.nq:
                    newid=str(newid)
                    toks = encode_single_newid(args, newid)
                    builder.add(toks)
            if args.tree == 1:
                root = builder.build()
            else:
                print('No Tree')
                root = None
            self.root = root
        ######

        self.args = args
        self.save_hyperparameters(args)
        # assert args.tie_word_embedding is not args.decode_embedding
        if args.decode_embedding:
            if self.args.position:
                expand_scale = args.max_output_length if not args.hierarchic_decode else 1
                self.decode_vocab_size = args.output_vocab_size * expand_scale + 2
            else:
                self.decode_vocab_size = 12
        else:
            self.decode_vocab_size = None

        t5_config = T5Config(
            num_layers=args.num_layers,
            num_decoder_layers=0 if args.softmax else args.num_decoder_layers,
            d_ff=args.d_ff,
            d_model=args.d_model,
            num_heads=args.num_heads,
            decoder_start_token_id=0,  # 1,
            output_past=True,
            d_kv=args.d_kv,
            dropout_rate=args.dropout_rate,
            decode_embedding=args.decode_embedding,
            hierarchic_decode=args.hierarchic_decode,
            decode_vocab_size=self.decode_vocab_size,
            output_vocab_size=args.output_vocab_size,
            tie_word_embeddings=args.tie_word_embedding,
            tie_decode_embedding=args.tie_decode_embedding,
            contrastive=args.contrastive,
            Rdrop=args.Rdrop,
            Rdrop_only_decoder=args.Rdrop_only_decoder,
            Rdrop_loss=args.Rdrop_loss,
            adaptor_decode=args.adaptor_decode,
            adaptor_efficient=args.adaptor_efficient,
            adaptor_layer_num = args.adaptor_layer_num,
            embedding_distillation=args.embedding_distillation,
            weight_distillation=args.weight_distillation,
            input_dropout=args.input_dropout,
            denoising=args.denoising,
            multiple_decoder=args.multiple_decoder,
            decoder_num=args.decoder_num,
            train_batch_size=args.train_batch_size,
            eval_batch_size=args.eval_batch_size,
            max_output_length=args.max_output_length,
        )
        print(t5_config)
        model = T5ForConditionalGeneration(t5_config)
        if args.pretrain_encoder:
            pretrain_model = T5ForConditionalGeneration.from_pretrained(args.model_name_or_path)
            pretrain_params = dict(pretrain_model.named_parameters())
            for name, param in model.named_parameters():
                if name.startswith(("shared.", "encoder.")):
                    with torch.no_grad():
                        param.copy_(pretrain_params[name])
        self.model = model
        print(self.model)
        self.tokenizer = T5Tokenizer.from_pretrained(args.tokenizer_name_or_path)
        # self.rouge_metric = load_metric('rouge')

        if self.args.freeze_embeds:
            self.freeze_embeds()
        if self.args.freeze_encoder:
            self.freeze_params(self.model.get_encoder())
            assert_all_frozen(self.model.get_encoder())
        if self.args.softmax:
            self.fc = torch.nn.Linear(args.d_model, self.args.num_cls)  # [feature size, num cls]
        self.ce = torch.nn.CrossEntropyLoss(ignore_index=-100)
        self.softmax = torch.nn.Softmax(dim=-1)
        self.triplet_loss = torch.nn.TripletMarginLoss(margin=1.0, p=2)
        self.ranking_loss = torch.nn.MarginRankingLoss(margin=0.5)
        if self.args.disc_loss:
            self.dfc = torch.nn.Linear(args.d_model, 1)

        n_observations_per_split = {
            "train": self.args.n_train,
            "validation": self.args.n_val,
            "test": self.args.n_test,
        }
        self.n_obs = {k: v if v >= 0 else None for k, v in n_observations_per_split.items()}

        if train:
            n_samples = self.n_obs['train']
            train_dataset = l1_query(self.args, self.tokenizer, n_samples)
            self.l1_query_train_dataset = train_dataset
            self.t_total = (
                    (len(train_dataset) // (self.args.train_batch_size * max(1, self.args.n_gpu)))
                    // self.args.gradient_accumulation_steps
                    * float(self.args.num_train_epochs)
            )

    def freeze_params(self, model):
        for par in model.parameters():
            par.requires_grad = False

    def freeze_embeds(self):
        """Freeze token embeddings and positional embeddings for bart, just token embeddings for t5."""
        try:
            self.freeze_params(self.model.model.shared)
            for d in [self.model.model.encoder, self.model.model.decoder]:
                self.freeze_params(d.embed_positions)
                self.freeze_params(d.embed_tokens)
        except AttributeError:
            self.freeze_params(self.model.shared)
            for d in [self.model.encoder, self.model.decoder]:
                self.freeze_params(d.embed_tokens)

    def lmap(self, f, x):
        """list(map(f, x))"""
        return list(map(f, x))

    def is_logger(self):
        return self.trainer.global_rank <= 0

    def parse_score(self, result):
        return {k: round(v.mid.fmeasure * 100, 4) for k, v in result.items()}

    def forward(self, input_ids, aug_input_ids=None, encoder_outputs=None, attention_mask=None, aug_attention_mask=None, logit_mask=None, decoder_input_ids=None, decoder_attention_mask=None, lm_labels=None, query_embedding=None, prefix_emb=None, prefix_mask=None, only_encoder=False, decoder_index=-1, input_mask=None):
        input_mask = None
        if self.args.Rdrop > 0 and not self.args.Rdrop_only_decoder and self.training:
            if aug_input_ids is not None and self.training:
                input_ids = torch.cat([input_ids, aug_input_ids.clone()], dim=0)
                attention_mask = torch.cat([attention_mask, aug_attention_mask], dim=0)
            elif self.training:
                input_ids = torch.cat([input_ids, input_ids.clone()], dim=0)
                attention_mask = torch.cat([attention_mask, attention_mask.clone()], dim=0)
            if self.args.denoising:
                if input_mask is None:
                    input_mask = torch.rand(input_ids.shape, device=input_ids.device) < 0.9
            if self.args.input_dropout and np.random.rand() < 0.5:
                if input_mask is None:
                    input_mask = torch.rand(input_ids.shape, device=input_ids.device) < 0.9
                input_ids = torch.where(input_mask==True, input_ids, torch.zeros_like(input_ids))
            if decoder_attention_mask is not None:
                decoder_attention_mask = torch.cat([decoder_attention_mask, decoder_attention_mask], dim=0)
            if lm_labels is not None:
                lm_labels = torch.cat([lm_labels, lm_labels], dim=0)
            if decoder_input_ids is not None:
                decoder_input_ids = torch.cat([decoder_input_ids, decoder_input_ids], dim=0)

        if self.args.loss_weight:
            loss_weight = torch.ones([input_ids.shape[0], self.args.max_output_length]).to(input_ids.device)
            loss_weight = loss_weight - torch.arange(start=0, end=0.5, step=0.5/self.args.max_output_length).reshape(1, -1).to(input_ids.device)
        else:
            loss_weight = None

        out = self.model(
            input_ids,
            input_mask=input_mask,
            logit_mask=logit_mask,
            encoder_outputs=encoder_outputs,
            only_encoder=only_encoder,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            lm_labels=lm_labels,
            query_embedding=query_embedding,
            prefix_embedding=prefix_emb,
            prefix_mask=prefix_mask,
            return_dict=True,
            output_hidden_states=True,
            decoder_index=decoder_index,
            loss_weight=loss_weight,
        )
        return out

    def _step(self, batch):
        loss, orig_loss, dist_loss, q_emb_distill_loss, weight_distillation = None, None, None, None, None
        if self.args.multiple_decoder:
            encoder_outputs, input_mask, generation_loss, denoising_loss = self.forward(input_ids=batch["source_ids"], aug_input_ids=batch["aug_source_ids"],
                                attention_mask=batch["source_mask"], aug_attention_mask=batch["aug_source_mask"],
                                query_embedding=batch["query_emb"], only_encoder=True)
            l1, l2, l3, l4, l5 = [], [], [], [], []
            for i in range(self.args.decoder_num):
                cl1, cl2, cl3, cl4, cl5 = self._step_i(batch, i, encoder_outputs=encoder_outputs, input_mask=input_mask)
                l1.append(cl1)
                l2.append(cl2)
                l3.append(cl3)
                l4.append(cl4)
                l5.append(cl5)
            loss = torch.stack(l1, dim=0).sum(dim=0) if l1[0] != 0 else 0
            orig_loss = torch.stack(l2, dim=0).sum(dim=0) if l2[0] != 0 else 0
            dist_loss = torch.stack(l3, dim=0).sum(dim=0) if l3[0] != 0 else 0
            q_emb_distill_loss = torch.stack(l4, dim=0).sum(dim=0) if l4[0] != 0 else 0
            weight_distillation = torch.stack(l5, dim=0).sum(dim=0) if l5[0] != 0 else 0
        else:
            loss, orig_loss, dist_loss, q_emb_distill_loss, weight_distillation = self._step_i(batch, -1)

        return loss, orig_loss, dist_loss, q_emb_distill_loss, weight_distillation

    def _step_i(self, batch, i, encoder_outputs=None, input_mask=None):
        if i < 0:
            lm_labels = batch["target_ids"]
            target_mask = batch['target_mask']
        else:
            lm_labels = batch["target_ids"][i]
            target_mask = batch['target_mask'][i]
        lm_labels[lm_labels[:, :] == self.tokenizer.pad_token_id] = -100

        outputs = self.forward(input_ids=batch["source_ids"], aug_input_ids=batch["aug_source_ids"],
                               attention_mask=batch["source_mask"], aug_attention_mask=batch["aug_source_mask"],
                               lm_labels=lm_labels, decoder_attention_mask=target_mask,
                               query_embedding=batch["query_emb"], decoder_index=i, encoder_outputs=encoder_outputs,
                               prefix_emb=batch["prefix_emb"], prefix_mask=batch["prefix_mask"], input_mask=input_mask)

        neg_outputs = None
        if self.args.hard_negative and self.args.sample_neg_num > 0:
            neg_lm_labels = torch.cat(batch['neg_target_ids'], dim=0)
            neg_decoder_attention_mask = torch.cat(batch['neg_target_mask'], dim=0)
            attention_mask = batch["source_mask"].repeat([self.args.sample_neg_num, 1])
            sources_ids = batch["source_ids"].repeat([self.args.sample_neg_num, 1])
            neg_lm_labels[neg_lm_labels[:, :] == self.tokenizer.pad_token_id] = -100
            neg_outputs = self.forward(input_ids=sources_ids, decoder_index=i, encoder_outputs=outputs.encoder_outputs, attention_mask=attention_mask, lm_labels=neg_lm_labels, decoder_attention_mask=neg_decoder_attention_mask, query_embedding=batch['query_emb'])

        def select_lm_head_weight(cur_outputs):
            lm_head_weight = cur_outputs.lm_head_weight
            vocab_size = lm_head_weight.shape[-1]
            dim_size = lm_head_weight.shape[-2]
            lm_head_weight = lm_head_weight.view(-1, vocab_size)    #[batch_size, seq_length, dim_size, vocab_size]
            indices = cur_outputs.labels.unsqueeze(-1).repeat([1, 1, dim_size]).view(-1, 1)
            indices[indices[:, :] == -100] = self.tokenizer.pad_token_id
            lm_head_weight = torch.gather(lm_head_weight, -1, indices)   #[batch_size, seq_length, dim_size, 1]
            lm_head_weight = lm_head_weight.view(cur_outputs.decoder_hidden_states[-1].shape)
            return lm_head_weight

        def cal_contrastive(outputs, neg_outputs):
            vocab_size = outputs.lm_head_weight.shape[-1]
            dim_size = outputs.lm_head_weight.shape[-2]
            decoder_weight = select_lm_head_weight(outputs)

            if neg_outputs is not None:
                decoder_embed = torch.cat((outputs.decoder_hidden_states[-1], neg_outputs.decoder_hidden_states[-1]), dim=0).transpose(0, 1).transpose(1, 2)        #[seq_length, embed_size, batch_size*2]
                neg_decoder_weight = select_lm_head_weight(neg_outputs)
                decoder_weight = torch.cat((decoder_weight, neg_decoder_weight), dim=0).transpose(0, 1).transpose(1, 2)
            else:
                decoder_embed = outputs.decoder_hidden_states[-1].transpose(0, 1).transpose(1, 2)
                decoder_weight = decoder_weight.transpose(0, 1).transpose(1, 2)
            seq_length = decoder_embed.shape[0]
            embed_size = decoder_embed.shape[1]
            bz = outputs.encoder_last_hidden_state.shape[0]
            #print("decoder_embed", decoder_embed.shape)  #[seq_length, embed_size, batch_size + neg_bz]
            #print("decoder_weight", decoder_weight.shape) #[seq_length, embed_size, batch_size + neg_bz]
            query_embed = outputs.encoder_last_hidden_state[:,0,:].unsqueeze(0).repeat([seq_length, 1, 1])  #[seq_length, batch_size, embed_size]
            #query_tloss = self.triplet_loss(query_embed, decoder_embed[:,:,0:bz], decoder_embed[:,:,bz:])
            #query_tloss = self.triplet_loss(query_embed, decoder_weight[:,:,0:bz], decoder_weight[:,:,bz:])
            query_tloss = None
            weight_tloss = None
            disc_loss = None
            ranking_loss = None
            if self.args.query_tloss:
                all_doc_embed = decoder_embed    #[seq_length, embed_size, pos_bz+neg_bz]
                doc_logits = torch.bmm(query_embed, all_doc_embed) #[sl, bz, bz+neg_bz]
                contrast_labels = torch.arange(0, bz).to(doc_logits.device).long()
                contrast_labels = contrast_labels.unsqueeze(0).repeat(seq_length, 1)
                #masks = outputs.labels.transpose(0, 1).repeat([1, 1+self.args.sample_neg_num])
                contrast_labels[outputs.labels.transpose(0, 1)[:, :] == -100] = -100
                query_tloss = self.ce(doc_logits.view(seq_length*bz, -1), contrast_labels.view(-1))
            if self.args.weight_tloss:
                query_embed = query_embed.transpose(1, 2)
                doc_embed = decoder_embed[:,:,0:bz].transpose(1, 2)  #[seq_length, batch_size, embed_size]
                query_logits = torch.bmm(doc_embed, query_embed) #[sl, bz, bz]
                contrast_labels = torch.arange(0, bz).to(query_logits.device).long()
                contrast_labels = contrast_labels.unsqueeze(0).repeat(seq_length, 1)   #[sl, bz]
                contrast_labels[outputs.labels.transpose(0, 1)[:, :] == -100] = -100
                weight_tloss = self.ce(query_logits.view(seq_length*bz, -1), contrast_labels.view(-1))
            if self.args.ranking_loss:
                rank_target = torch.ones(bz*seq_length).to(lm_labels.device)
                rank_indices = outputs.labels.detach().clone().reshape([-1, 1])
                rank_indices[rank_indices[:, :] == -100] = self.tokenizer.pad_token_id
                #pos_prob = torch.gather(self.softmax(outputs.lm_logits.detach().clone()).view(-1, vocab_size), -1, rank_indices).squeeze(-1)
                pos_prob = torch.gather(self.softmax(outputs.lm_logits).view(-1, vocab_size), -1, rank_indices)
                pos_prob[rank_indices[:,:] == self.tokenizer.pad_token_id] = 1.0
                pos_prob = pos_prob.squeeze(-1)
                #[bz, seq_length, vocab_size] -> [bz, seq_length]
                #pos_prob, _ = torch.max(self.softmax(outputs.lm_logits.detach()), -1)
                neg_prob, _ = torch.max(self.softmax(neg_outputs.lm_logits), -1)
                ranking_loss = self.ranking_loss(pos_prob.view(-1), neg_prob.view(-1), rank_target)
            if self.args.disc_loss:
                target = torch.zeros(seq_length, bz).to(lm_labels.device)
                target[outputs.labels.transpose(0, 1)[:, :] == -100] = -100
                all_logits = self.dfc(torch.reshape(decoder_embed.transpose(1,2),(-1, embed_size))).view(seq_length, -1)  #[seq_length, bz+neg_bz]
                all_logits = all_logits.view(seq_length, self.args.sample_neg_num+1, bz).transpose(1, 2)
                all_logits = torch.reshape(all_logits, (-1, self.args.sample_neg_num+1)) #[seq_length*bz, pos+neg_num]
                disc_loss = self.ce(all_logits.view(-1, self.args.sample_neg_num+1), target.view(-1).long())
            return query_tloss, weight_tloss, disc_loss, ranking_loss

        if self.args.softmax:
            logits = self.fc(outputs.encoder_last_hidden_state)[:, 0, :].squeeze()
            loss = self.ce(logits, batch["softmax_index"].squeeze(dim=1))
        else:
            if self.args.hard_negative:
                query_tloss, weight_tloss, disc_loss, ranking_loss = cal_contrastive(outputs, neg_outputs)
                loss = outputs.loss
                if self.args.ranking_loss:
                    loss += ranking_loss
                if self.args.disc_loss:
                    loss += disc_loss
                    loss = outputs.loss
                if self.args.query_tloss:
                    loss += query_tloss
                if self.args.weight_tloss:
                    loss += weight_tloss
            else:
                loss = outputs.loss

        if self.args.Rdrop > 0:
            orig_loss = outputs.orig_loss
            dist_loss = outputs.dist_loss
        else:
            orig_loss, dist_loss = 0,0

        if self.args.embedding_distillation > 0:
            q_emb_distill_loss = outputs.emb_distill_loss
        else:
            q_emb_distill_loss = 0


        if self.args.weight_distillation > 0:
            weight_distillation = outputs.weight_distillation
        else:
            weight_distillation = 0

        return loss, orig_loss, dist_loss, q_emb_distill_loss, weight_distillation


    def _softmax_generative_step(self, batch):
        assert self.args.softmax
        lm_labels = batch["target_ids"]
        lm_labels[lm_labels[:, :] == self.tokenizer.pad_token_id] = -100

        outputs = self.forward(
            input_ids=batch["source_ids"],
            attention_mask=batch["source_mask"],
            lm_labels=lm_labels,
            decoder_attention_mask=batch['target_mask'],
        )

        pred_index = torch.argmax(outputs[0], dim=1)
        return pred_index


    def ids_to_clean_text(self, generated_ids):
        gen_text = self.tokenizer.batch_decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )
        return self.lmap(str.strip, gen_text)


    def training_step(self, batch, batch_idx):
        # set to train
        loss, orig_loss, kl_loss, q_emb_distill_loss, weight_distillation = self._step(batch)
        self.log("train_loss", loss)
        return {"loss":loss, "orig_loss":orig_loss, "kl_loss":kl_loss,
                "Query_distill_loss":q_emb_distill_loss,
                "Weight_distillation":weight_distillation}


    def training_epoch_end(self, outputs):
        avg_train_loss = torch.stack([x["loss"] for x in outputs]).mean()
        self.log("avg_train_loss", avg_train_loss)

    def validation_step(self, batch, batch_idx):
        if self.args.multiple_decoder:
            result_list = []
            for i in range(self.args.decoder_num):
                result = self.validation_step_i(batch, i)
                result_list.append(result)
            return result_list
        else:
            result = self.validation_step_i(batch, -1)
            return result

    def validation_step_i(self, batch, i):

        inf_result_cache = []
        if self.args.decode_embedding:
            if self.args.position:
                expand_scale = self.args.max_output_length if not self.args.hierarchic_decode else 1
                decode_vocab_size = self.args.output_vocab_size * expand_scale + 2
            else:
                decode_vocab_size = 12
        else:
            decode_vocab_size = None

        assert not self.args.softmax and self.args.gen_method == "greedy"

        if self.args.decode_embedding == 1:
            outs, scores = self.model.generate(
                batch["source_ids"].cuda(),
                attention_mask=batch["source_mask"].cuda(),
                use_cache=False,
                decoder_attention_mask=batch['target_mask'],
                max_length=self.args.max_output_length,
                num_beams=self.args.num_return_sequences,
                length_penalty=self.args.length_penalty,
                num_return_sequences=self.args.num_return_sequences,
                early_stopping=False, 
                decode_embedding=self.args.decode_embedding,
                decode_vocab_size=decode_vocab_size,
                output_scores=True
            )
            dec = [numerical_decoder(self.args, ids, output=True) for ids in outs]
        elif self.args.decode_embedding == 2:
            if self.args.multiple_decoder:
                target_mask = batch['target_mask'][i].cuda()
            else:
                target_mask = batch['target_mask'].cuda()
            outs, scores = self.model.generate(
                batch["source_ids"].cuda(),
                attention_mask=batch["source_mask"].cuda(),
                use_cache=False,
                decoder_attention_mask=target_mask,
                max_length=self.args.max_output_length,
                num_beams=self.args.num_return_sequences,
                length_penalty=self.args.length_penalty,
                num_return_sequences=self.args.num_return_sequences,
                early_stopping=False,  
                decode_embedding=self.args.decode_embedding,
                decode_vocab_size=decode_vocab_size,
                decode_tree=self.root,
                decoder_index=i,
                output_scores=True
            )
            dec = decode_token(self.args, outs.cpu().numpy())  # num = 10*len(pred)
        else:
            outs, scores = self.model.generate(
                batch["source_ids"].cuda(),
                attention_mask=batch["source_mask"].cuda(),
                use_cache=False,
                decoder_attention_mask=batch['target_mask'],
                max_length=self.args.max_output_length,
                num_beams=self.args.num_return_sequences,
                # no_repeat_ngram_size=2,
                length_penalty=self.args.length_penalty,
                num_return_sequences=self.args.num_return_sequences,
                early_stopping=False,  # False,
                decode_embedding=self.args.decode_embedding,
                decode_vocab_size=decode_vocab_size,
                decode_tree=self.root,
                output_scores=True
            )
            dec = [self.tokenizer.decode(ids) for ids in outs]

        texts = [self.tokenizer.decode(ids) for ids in batch['source_ids']]

        dec = dec_2d(dec, self.args.num_return_sequences)
        for r in batch['rank']:
            if self.args.label_length_cutoff:
                gt = [s[:self.args.max_output_length - 2] for s in list(r[0])]
            else:
                gt = list(r[0])
            ra = r[1]
            ra = [str(a.item()) for a in ra]

            for pred, g, text, ran in zip(dec, gt, texts, ra):
                pred = ','.join(pred)
                inf_result_cache.append([text, pred, g, int(ran)])
        return {"inf_result_batch": inf_result_cache, 'inf_result_batch_prob': scores}

    def validation_epoch_end(self, outputs):  
        if self.args.multiple_decoder:
            reverse_outputs = []
            for j in range(len(outputs[0])):
                reverse_outputs.append([])
            for i in range(len(outputs)):
                for j in range(len(outputs[0])):
                    reverse_outputs[j].append(outputs[i][j])
            outputs = reverse_outputs

        if self.args.multiple_decoder:
            inf_result_cache = []
            inf_result_cache_prob = []
            for index in range(self.args.decoder_num):
                cur_inf_result_cache = [item for sublist in outputs[index] for item in sublist['inf_result_batch']]
                cur_inf_result_cache_prob = [softmax(sublist['inf_result_batch_prob'][i * int(len(sublist['inf_result_batch_prob'])/len(outputs[index][0]['inf_result_batch'])): (i + 1) * int(len(sublist['inf_result_batch_prob'])/len(outputs[index][0]['inf_result_batch']))]) for sublist in outputs[index] for i in range(len(sublist['inf_result_batch']))]
                inf_result_cache.extend(cur_inf_result_cache)
                inf_result_cache_prob.extend(cur_inf_result_cache_prob)
        else:
            inf_result_cache = [item for sublist in outputs for item in sublist['inf_result_batch']]
            inf_result_cache_prob = [softmax(sublist['inf_result_batch_prob'][i * int(len(sublist['inf_result_batch_prob'])/len(outputs[0]['inf_result_batch'])): (i + 1) * int(len(sublist['inf_result_batch_prob'])/len(outputs[0]['inf_result_batch']))]) for sublist in outputs for i in range(len(sublist['inf_result_batch']))]

        res = pd.DataFrame(inf_result_cache, columns=["query", "pred", "gt", "rank"])
        res.sort_values(by=['query', 'rank'], ascending=True, inplace=True)
        res1 = res.loc[res['rank'] == 1]
        res1 = res1.values.tolist()

        if self.args.trivia:
            q_gt, q_pred = {}, {}
            prev_q = ""
            for [query, pred, gt, _] in res1:
                if query != prev_q:
                    q_pred[query] = pred.split(",")
                    q_pred[query] = q_pred[query][:5]
                    q_pred[query] = list(set(q_pred[query]))
                    prev_q = query
                if query in q_gt:
                    if len(q_gt[query]) <= 100:
                        q_gt[query].add(gt)
                else:
                    q_gt[query] = gt.split(",")
                    q_gt[query] = set(q_gt[query])
        else:
            q_gt, q_pred = {}, {}
            prev_q = ""
            for [query, pred, gt, _] in res1:
                if query != prev_q:
                    pred_list = pred.split(",")
                    if query not in q_pred:
                        q_pred[query] = set(pred_list[:1])
                    else:
                        q_pred[query].add(pred_list[0])
                    prev_q = query
                else:
                    pred_list = pred.split(",")
                    if query not in q_pred:
                        q_pred[query] = set(pred_list[:1])
                    else:
                        q_pred[query].add(pred_list[0])
                    prev_q = query
                if query in q_gt:
                    if len(q_gt[query]) <= 100:
                        q_gt[query].add(gt)
                else:
                    q_gt[query] = set()
                    q_gt[query].add(gt)

        total = 0
        for q in q_pred:
            print(q, q_pred[q], q_gt[q])
            is_hit = 0
            for p in q_gt[q]:
                if p in q_pred[q]:
                    is_hit = 1
            total += is_hit
        recall_avg = total / len(q_pred)
        print("recall@{}:{}".format(self.args.decoder_num, recall_avg))
        self.log("recall1", recall_avg)


    def configure_optimizers(self):
        "Prepare optimizer and schedule (linear warmup and decay)"

        model = self.model
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if
                           (not any(nd in n for nd in no_decay)) and (n.startswith(("shared.", "encoder.")))],
                "weight_decay": self.args.weight_decay,
                "lr": self.args.learning_rate,
            },
            {
                "params": [p for n, p in model.named_parameters() if
                           (not any(nd in n for nd in no_decay)) and (not n.startswith(("shared.", "encoder.")))],
                "weight_decay": self.args.weight_decay,
                "lr": self.args.decoder_learning_rate,
            },
            {
                "params": [p for n, p in model.named_parameters() if
                           (any(nd in n for nd in no_decay)) and (n.startswith(("shared.", "encoder.")))],
                "weight_decay": 0.0,
                "lr": self.args.learning_rate,
            },
            {
                "params": [p for n, p in model.named_parameters() if
                           (any(nd in n for nd in no_decay)) and (not n.startswith(("shared.", "encoder.")))],
                "weight_decay": 0.0,
                "lr": self.args.decoder_learning_rate,
            },
        ]
        optimizer = AdamW(optimizer_grouped_parameters, eps=self.args.adam_epsilon)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=self.args.warmup_steps, num_training_steps=self.t_total
        )
        return [optimizer], [{"scheduler": scheduler, "interval": "step", "frequency": 1}]


    def get_tqdm_dict(self):
        tqdm_dict = {"loss": "{:.3f}".format(self.trainer.avg_loss), "lr": self.lr_scheduler.get_last_lr()[-1]}
        return tqdm_dict


    def train_dataloader(self):
        print('load training data and create training loader.')
        n_samples = self.n_obs['train']
        if hasattr(self, 'l1_query_train_dataset'):
            train_dataset = self.l1_query_train_dataset
        else:
            train_dataset = l1_query(self.args, self.tokenizer, n_samples)
        self.prefix_embedding, self.prefix2idx_dict, self.prefix_mask = \
            train_dataset.prefix_embedding, train_dataset.prefix2idx_dict, train_dataset.prefix_mask
        sampler = DistributedSampler(train_dataset)
        dataloader = DataLoader(train_dataset, sampler=sampler, batch_size=self.args.train_batch_size,
                                drop_last=True, shuffle=False, num_workers=4)
        return dataloader


    def val_dataloader(self):
        print('load validation data and create validation loader.')
        n_samples = self.n_obs['validation']
        val_dataset = l1_query(self.args, self.tokenizer, n_samples, task='test')
        sampler = DistributedSampler(val_dataset)
        dataloader = DataLoader(val_dataset, sampler=sampler, batch_size=self.args.eval_batch_size,
                                drop_last=True, shuffle=False, num_workers=4)
        return dataloader
