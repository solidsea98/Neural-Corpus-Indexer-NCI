cd kmeans
nohup python -u kmeans.py --bert_size 512 --k 30 --c 30 --dataset Trivia > Trivia_kmeans_512.log 2>&1 &
