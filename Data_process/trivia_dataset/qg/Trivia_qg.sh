cd qg
nohup python -u qg.py --idx 0 --cuda_device 0 --return_num 15 --max_len 64 --dataset Trivia > log/Trivia_0.log 2>&1 &
nohup python -u qg.py --idx 1 --cuda_device 1 --return_num 15 --max_len 64 --dataset Trivia > log/Trivia_1.log 2>&1 &
nohup python -u qg.py --idx 2 --cuda_device 2 --return_num 15 --max_len 64 --dataset Trivia > log/Trivia_2.log 2>&1 &
nohup python -u qg.py --idx 3 --cuda_device 3 --return_num 15 --max_len 64 --dataset Trivia > log/Trivia_3.log 2>&1 &
nohup python -u qg.py --idx 4 --cuda_device 4 --return_num 15 --max_len 64 --dataset Trivia > log/Trivia_4.log 2>&1 &
nohup python -u qg.py --idx 5 --cuda_device 5 --return_num 15 --max_len 64 --dataset Trivia > log/Trivia_5.log 2>&1 &
nohup python -u qg.py --idx 6 --cuda_device 6 --return_num 15 --max_len 64 --dataset Trivia > log/Trivia_6.log 2>&1 &
nohup python -u qg.py --idx 7 --cuda_device 7 --return_num 15 --max_len 64 --dataset Trivia > log/Trivia_7.log 2>&1 &
