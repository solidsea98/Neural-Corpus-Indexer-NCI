cd qg
nohup python -u nq_qg.py --idx 0 --cuda_device 0 --return_num 20 --max_len 64 > nq_0.log 2>&1 &
nohup python -u nq_qg.py --idx 1 --cuda_device 1 --return_num 20 --max_len 64 > nq_1.log 2>&1 &
nohup python -u nq_qg.py --idx 2 --cuda_device 2 --return_num 20 --max_len 64 > nq_2.log 2>&1 &
nohup python -u nq_qg.py --idx 3 --cuda_device 3 --return_num 20 --max_len 64 > nq_3.log 2>&1 &
nohup python -u nq_qg.py --idx 4 --cuda_device 4 --return_num 20 --max_len 64 > nq_4.log 2>&1 &
nohup python -u nq_qg.py --idx 5 --cuda_device 5 --return_num 20 --max_len 64 > nq_5.log 2>&1 &
nohup python -u nq_qg.py --idx 6 --cuda_device 6 --return_num 20 --max_len 64 > nq_6.log 2>&1 &
nohup python -u nq_qg.py --idx 7 --cuda_device 7 --return_num 20 --max_len 64 > nq_7.log 2>&1 &
