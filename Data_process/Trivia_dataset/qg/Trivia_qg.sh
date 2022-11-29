# +
cd qg
mkdir pkl
mkdir log
PARTITION_NUM=$1

curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo bash
sudo apt-get install git-lfs
git clone https://huggingface.co/castorini/doc2query-t5-base-msmarco


for ITER in $(seq 0 $PARTITION_NUM)
do
nohup python -u qg.py --idx $ITER --cuda_device $ITER --return_num 15 --max_len 64 --dataset Trivia > log/Trivia_$ITER.log 2>&1 &
done
