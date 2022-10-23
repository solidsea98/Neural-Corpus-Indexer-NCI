from sklearn.cluster import KMeans
from sklearn.cluster import MiniBatchKMeans
import numpy as np
import pandas as pd
import pickle
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--v_dim', type=int, default=768)
parser.add_argument('--dataset', type=str, default='NQ')
parser.add_argument('--bert_size', type=int, default=512)
parser.add_argument('--seed', type=int, default=7)
parser.add_argument('--k', type=int, default= 10)
parser.add_argument('--c', type=int, default= 100)

args = parser.parse_args()


df = pd.read_csv(f'../bert/{args.dataset}_doc_content_embedding_bert_{args.bert_size}.tsv',
                 names=['docid', 'url', 'title', 'body', 'anchor', 'click', 'language', 'vector'],
                 header=None, sep='\t').loc[:, ['docid', 'vector']]
df.drop_duplicates('docid', inplace = True)
old_id = df['docid'].tolist()
X = df['vector'].tolist()
for idx,v in enumerate(X):
    vec_str = v.split('|')
    if len(vec_str) != args.v_dim:
        print('vec dim error!')
        exit(1)
    X[idx] = [float(v) for v in vec_str]
X = np.array(X)
print(X.shape)
new_id_list = []

kmeans = KMeans(n_clusters=args.k, max_iter=300, n_init=100, init='k-means++', random_state=args.seed, tol=1e-7)

mini_kmeans = MiniBatchKMeans(n_clusters=args.k, max_iter=300, n_init=100, init='k-means++', random_state=3,
                              batch_size=1000, reassignment_ratio=0.01, max_no_improvement=20, tol=1e-7)


def classify_recursion(x_data_pos):
    if x_data_pos.shape[0] <= args.c:
        if x_data_pos.shape[0] == 1:
            return
        for idx, pos in enumerate(x_data_pos):
            new_id_list[pos].append(idx)
        return

    temp_data = np.zeros((x_data_pos.shape[0], args.v_dim))
    for idx, pos in enumerate(x_data_pos):
        temp_data[idx, :] = X[pos]

    if x_data_pos.shape[0] >= 1e3:
        pred = mini_kmeans.fit_predict(temp_data)
    else:
        pred = kmeans.fit_predict(temp_data)

    for i in range(args.k):
        pos_lists = []
        for id_, class_ in enumerate(pred):
            if class_ == i:
                pos_lists.append(x_data_pos[id_])
                new_id_list[x_data_pos[id_]].append(i)
        classify_recursion(np.array(pos_lists))

    return

print('Start First Clustering')
pred = mini_kmeans.fit_predict(X)
print(pred.shape)   #int 0-9 for each vector
print(mini_kmeans.n_iter_)

for class_ in pred:
    new_id_list.append([class_])

print('Start Recursively Clustering...')
for i in range(args.k):
    print(i, "th cluster")
    pos_lists = [];
    for id_, class_ in enumerate(pred):
        if class_ == i:
            pos_lists.append(id_)
    classify_recursion(np.array(pos_lists))

#print(new_id_list[100:200])
mapping = {}
for i in range(len(old_id)):
    mapping[old_id[i]] = new_id_list[i]

with open(f'IDMapping_{args.dataset}_bert_{args.bert_size}_k{args.k}_c{args.c}_seed_{args.seed}.pkl', 'wb') as f:
    pickle.dump(mapping, f)
