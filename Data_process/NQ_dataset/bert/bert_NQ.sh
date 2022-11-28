# +
cd bert
mkdir pkl
mkdir log
MAX_LEN=512
PARTITION_NUM=$1

for ITER in $(seq 0 $PARTITION_NUM)
do
nohup python -u bert.py --dataset NQ --partition_num ${PARTITION_NUM} --idx ${ITER} --cuda_device ${ITER} --max_len ${MAX_LEN} > log/NQ_bert_512_${ITER}.log 2>&1 &
done
