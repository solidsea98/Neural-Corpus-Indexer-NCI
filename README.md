# A Neural Corpus Indexer for Document Retrieval (NCI)

[![made-with-python](https://img.shields.io/badge/Made%20with-Python3-1f425f.svg?color=purple)](https://www.python.org/)

## What is NCI?

**NCI** is an end-to-end, sequence-to-sequence differentiable **document retrieval model** which retrieve relevant document identifiers directly for specific queries. In our evaluation on [Google NQ dataset](https://ai.google.com/research/NaturalQuestions/) and [TriviaQA dataset](https://nlp.cs.washington.edu/triviaqa/), NCI outperforms all baselines and model-based indexers:

| Model | Recall@1 | Recall@10 | Recall@100 | MRR@100 |
|:-------:|:--------:|:-----:|:-----:|:-------:|
| **NCI (Ensemble)** | **70.46** | **89.35** | **94.75** | **77.82** |
| **NCI (Large)** | **66.23** | **85.27** | **92.49** | **73.37** |
| **NCI (Base)** | **65.86** | **85.20** | **92.42** | **73.12** |
| DSI (T5-Base) | 27.40 | 56.60 | -- | -- |
| DSI (T5-Large) | 35.60 | 62.60 | -- | -- |
| SEAL (Large) | 59.93 | 81.24 | 90.93 | 67.70 |
| ANCE (MaxP) | 52.63 | 80.38 | 91.31 | 62.84 |
| BM25 + DocT5Query | 35.43 | 61.83 | 76.92 | 44.47 |

For more information, checkout our publications: 
*https://arxiv.org/abs/2206.02743*

<!-- Cite as below if you find this repository is helpful to your project:

```
Wang Y, Hou Y, Wang H, et al. A Neural Corpus Indexer for Document Retrieval[J]. arXiv preprint arXiv:2206.02743, 2022.
``` -->

## Environemnt
[1] Install [Anaconda](https://www.anaconda.com/download).

[2] Clone repository:

```bash
git clone https://github.com/solidsea98/Neural-Corpus-Indexer-NCI.git
cd Neural-Corpus-Indexer-NCI
```

[3] Create conda environment:

```bash
conda env create -f environment.yml
conda activate NCI
```
[4] Docker:

If necessary, the NCI docker is mzmssg/corpus_env:latest.

## Data Process

You can process data with [NQ_dataset_Process.ipynb](./Data_process/NQ_dataset/NQ_dataset_Process.ipynb) and [Trivia_dataset_Process.ipynb](./Data_process/trivia_dataset/Trivia_dataset_Process.ipynb).

### Dataset Download.

Currently NCI is evaluated on [Google NQ dataset](https://ai.google.com/research/NaturalQuestions) and [TriviaQA dataset](https://nlp.cs.washington.edu/triviaqa/data/triviaqa-rc.tar.gz). Please download it before re-training.


### Semantic Identifier

NCI uses content-based document identifiers: A pre-trained BERT is used to generate document embeddings, and then documents are clustered using hierarchical K-means and semantic identifiers are assigned to each document. You can generate several embeddings and semantic identifiers to run NCI model for ensembling.

Please find more details in [NQ_dataset_Process.ipynb](./Data_process/NQ_dataset/NQ_dataset_Process.ipynb) and [Trivia_dataset_Process.ipynb](./Data_process/trivia_dataset/Trivia_dataset_Process.ipynb).


### Query Generation

In our study, Query Generation can significantly improve retrieve performance, especially for long-tail queries.

NCI uses [docTTTTTquery](https://github.com/castorini/docTTTTTquery) checkpoint to generate synthetic queries. Please refer to docTTTTTquery documentation and find more details in [NQ_dataset_Process.ipynb](./Data_process/NQ_dataset/NQ_dataset_Process.ipynb) and [Trivia_dataset_Process.ipynb](./Data_process/Trivia_dataset/Trivia_dataset_Process.ipynb).


## Training

Once the data pre-processing is complete, you can launch training by [train.sh](./NCI_model/train.sh). You can also launch training along with our [NQ data](https://drive.google.com/drive/folders/1epfUw4yQjAtqnZTQDLAUOwTJg-YMCGdD?usp=sharing) (Download it to './Data_process/NQ_dataset/') and [TriviaQA data](https://drive.google.com/drive/folders/1_abDsHRUQabwDmBM7sk_NyMuia5X_VMK?usp=sharing) (Download it to './Data_process/trivia_dataset/').


## Evaluation
Please use [infer.sh](./NCI_model/infer.sh) along with our [NQ checkpoint](https://drive.google.com/file/d/1SITW9d7XLai6wSvu_f_8AYz38c7FQOBB/view?usp=sharing) or [TriviaQA checkpoint](https://drive.google.com/file/d/1XCA-XMDIZAZnlqecZrXzurKoZe7CzQhO/view?usp=sharing) (Download it to './NCI_model/logs/'). You can also inference with your own checkpoint to evaluate model performance.

Please ensemble [NQ dataset](./NCI_model/ensemble_NQ.ipynb) or [TriviaQA dataset](./NCI_model/ensemble_trivia.ipynb) along with [our results](https://drive.google.com/drive/folders/14TN0lEKHMh5eB5CBTWUp8SSwggiRXex3?usp=sharing) (Download it to './NCI_model/logs/') or your own results.


## Acknowledgement

We learned a lot and borrowed some code from the following projects when building NCI.
- [Transformers](https://github.com/huggingface/transformers)
- [docTTTTTquery](https://github.com/castorini/docTTTTTquery) 
