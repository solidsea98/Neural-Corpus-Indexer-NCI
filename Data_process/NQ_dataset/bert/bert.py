from transformers import BertTokenizer, BertModel
from tqdm import tqdm
import argparse
import pickle
import pandas as pd

def main(args):
    id_doc_dict = {}
    train_file = "../NQ_doc_content.tsv"

    with open(train_file, 'r') as f:
        for line in f.readlines():
            docid, _, _, content, _, _, _ = line.split("\t")
            id_doc_dict[docid] = content

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained("bert-base-uncased").to(f'cuda:{args.cuda_device}')

    ids = list(id_doc_dict.keys())
    text_list_all = []
    text_id_all = []
    i = 0
    batch_size = 40

    while (i < int(len(ids) / batch_size) - 1):
        id_list = ids[i * batch_size: (i + 1) * batch_size]
        text_list = [id_doc_dict[id_] for id_ in id_list]
        text_list_all.append(text_list)
        text_id_all.append(id_list)
        i += 1

    id_list = ids[i * batch_size:]
    text_list = [id_doc_dict[id_] for id_ in id_list]
    text_list_all.append(text_list)
    text_id_all.append(id_list)

    text_partitation = []
    text_partitation_id = []

    base = int(len(text_list_all) / 8)

    text_partitation.append(text_list_all[:base])
    text_partitation_id.append(text_id_all[:base])

    text_partitation.append(text_list_all[base: 2 * base])
    text_partitation_id.append(text_id_all[base: 2 * base])

    text_partitation.append(text_list_all[2 * base: 3 * base])
    text_partitation_id.append(text_id_all[2 * base: 3 * base])

    text_partitation.append(text_list_all[3 * base: 4 * base])
    text_partitation_id.append(text_id_all[3 * base: 4 * base])

    text_partitation.append(text_list_all[4 * base: 5 * base])
    text_partitation_id.append(text_id_all[4 * base: 5 * base])

    text_partitation.append(text_list_all[5 * base: 6 * base])
    text_partitation_id.append(text_id_all[5 * base: 6 * base])

    text_partitation.append(text_list_all[6 * base: 7 * base])
    text_partitation_id.append(text_id_all[6 * base: 7 * base])

    text_partitation.append(text_list_all[7 * base:  ])
    text_partitation_id.append(text_id_all[7 * base:  ])


    output_tensor = []
    output_id_tensor = []
    count = 0

    for elem in tqdm(text_partitation[args.idx]):
        encoded_input = tokenizer(elem, max_length=args.max_len, return_tensors='pt', padding=True, truncation=True).to(
            f'cuda:{args.cuda_device}')
        output = model(**encoded_input).last_hidden_state.detach().cpu()[:, 0, :].numpy().tolist()
        output_tensor.extend(output)
        output_id_tensor.extend(text_partitation_id[args.idx][count])
        count += 1

    output = open(f'nq_outpt_tensor_{args.max_len}_content_{args.idx}.pkl', 'wb', -1)
    pickle.dump(output_tensor, output)
    output.close()

    output = open(f'nq_outpt_tensor_{args.max_len}_content_{args.idx}_id.pkl', 'wb', -1)
    pickle.dump(output_id_tensor, output)
    output.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Options for Commonsense Knowledge Base Completion')

    parser.add_argument("--idx", type=int, default=0, help="partitation")
    parser.add_argument("--cuda_device", type=int, default=0, help="cuda")
    parser.add_argument("--max_len", type=int, default=512, help="cuda")

    args = parser.parse_args()
    print(args)

    main(args)
