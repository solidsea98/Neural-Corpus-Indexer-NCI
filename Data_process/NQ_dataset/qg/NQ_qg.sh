# +
cd qg
mkdir pkl
mkdir log
PARTITION_NUM=$1

for ITER in $(seq 0 $PARTITION_NUM)
do
nohup python -u qg.py --idx $ITER --cuda_device $ITER --return_num 15 --max_len 64 --dataset NQ > log/NQ_$ITER.log 2>&1 &
done
