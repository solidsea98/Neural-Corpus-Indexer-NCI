cd bert
MAX_LEN=512
nohup python -u bert.py --idx 0 --cuda_device 0 --max_len $MAX_LEN > bert_512_0.log 2>&1 &
nohup python -u bert.py --idx 1 --cuda_device 1 --max_len $MAX_LEN > bert_512_1.log 2>&1 &
nohup python -u bert.py --idx 2 --cuda_device 2 --max_len $MAX_LEN > bert_512_2.log 2>&1 &
nohup python -u bert.py --idx 3 --cuda_device 3 --max_len $MAX_LEN > bert_512_3.log 2>&1 &
nohup python -u bert.py --idx 4 --cuda_device 4 --max_len $MAX_LEN > bert_512_4.log 2>&1 &
nohup python -u bert.py --idx 5 --cuda_device 5 --max_len $MAX_LEN > bert_512_5.log 2>&1 &
nohup python -u bert.py --idx 6 --cuda_device 6 --max_len $MAX_LEN > bert_512_6.log 2>&1 &
nohup python -u bert.py --idx 7 --cuda_device 7 --max_len $MAX_LEN > bert_512_7.log 2>&1 &
