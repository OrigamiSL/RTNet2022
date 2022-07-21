# End-to-end
python -u main.py --model RT --data ETTh1 --features S  --label_len 168  --pred_len 24 --pyramid 3 --block_nums 3 --time_nums 2 --kernel 3 --learning_rate 0.0001 --dropout 0.1 --criterion Standard --test_inverse --d_model 32 --timebed None --batch_size 16 --itr 20 --reproducible

python -u main.py --model RT --data ETTh1 --features S  --label_len 168  --pred_len 48 --pyramid 3 --block_nums 3 --time_nums 2 --kernel 3 --learning_rate 0.0001 --dropout 0.1 --criterion Standard --test_inverse --d_model 32 --timebed None --batch_size 16 --itr 20 --reproducible

python -u main.py --model RT --data ETTh1 --features S  --label_len 168  --pred_len 168 --pyramid 3 --block_nums 3 --time_nums 2 --kernel 3 --learning_rate 0.0001 --dropout 0.1 --criterion Standard --test_inverse --d_model 32 --timebed None --batch_size 16 --itr 20 --reproducible

python -u main.py --model RT --data ETTh1 --features S  --label_len 168  --pred_len 336 --pyramid 3 --block_nums 3 --time_nums 2 --kernel 3 --learning_rate 0.0001 --dropout 0.1 --criterion Standard --test_inverse --d_model 32 --timebed None --batch_size 16 --itr 20 --reproducible

python -u main.py --model RT --data ETTh1 --features S  --label_len 168  --pred_len 720 --pyramid 3 --block_nums 3 --time_nums 2 --kernel 3 --learning_rate 0.0001 --dropout 0.1 --criterion Standard --test_inverse --d_model 32 --timebed None --batch_size 16 --itr 20 --reproducible

python -u main.py --model RT --data ETTh1 --features M  --label_len 168  --pred_len 24 --pyramid 3 --block_nums 3 --time_nums 2 --kernel 3 --learning_rate 0.0001 --dropout 0.1 --criterion Standard --test_inverse --d_model 112 --timebed None --group --angle 45 --itr 20 --reproducible

python -u main.py --model RT --data ETTh1 --features M  --label_len 168  --pred_len 48 --pyramid 3 --block_nums 3 --time_nums 2 --kernel 3 --learning_rate 0.0001 --dropout 0.1 --criterion Standard --test_inverse --d_model 112 --timebed None --group --angle 45 --itr 20 --reproducible

python -u main.py --model RT --data ETTh1 --features M  --label_len 168  --pred_len 168 --pyramid 3 --block_nums 3 --time_nums 2 --kernel 3 --learning_rate 0.0001 --dropout 0.1 --criterion Standard --test_inverse --d_model 112 --timebed None --group --angle 45 --itr 20 --reproducible

python -u main.py --model RT --data ETTh1 --features M  --label_len 168  --pred_len 336 --pyramid 3 --block_nums 3 --time_nums 2 --kernel 3 --learning_rate 0.0001 --dropout 0.1 --criterion Standard --test_inverse --d_model 112 --timebed None --group --angle 45 --itr 20 --reproducible

python -u main.py --model RT --data ETTh1 --features M  --label_len 168  --pred_len 720 --pyramid 3 --block_nums 3 --time_nums 2 --kernel 3 --learning_rate 0.0001 --dropout 0.1 --criterion Standard --test_inverse --d_model 112 --timebed None --group --angle 45 --itr 20 --reproducible

python -u main.py --model RT --data ETTh2 --features S  --label_len 168  --pred_len 24 --pyramid 3 --block_nums 3 --time_nums 2 --kernel 3 --learning_rate 0.0001 --dropout 0.1 --criterion Standard --test_inverse --d_model 64 --timebed None --itr 20 --reproducible

python -u main.py --model RT --data ETTh2 --features S  --label_len 168  --pred_len 48 --pyramid 3 --block_nums 3 --time_nums 2 --kernel 3 --learning_rate 0.0001 --dropout 0.1 --criterion Standard --test_inverse --d_model 64 --timebed None --itr 20 --reproducible

python -u main.py --model RT --data ETTh2 --features S  --label_len 168  --pred_len 168 --pyramid 3 --block_nums 3 --time_nums 2 --kernel 3 --learning_rate 0.0001 --dropout 0.1 --criterion Standard --test_inverse --d_model 64 --timebed None --itr 20 --reproducible

python -u main.py --model RT --data ETTh2 --features S  --label_len 168  --pred_len 336 --pyramid 3 --block_nums 3 --time_nums 2 --kernel 3 --learning_rate 0.0001 --dropout 0.1 --criterion Standard --test_inverse --d_model 64 --timebed None --itr 20 --reproducible

python -u main.py --model RT --data ETTh2 --features S  --label_len 168  --pred_len 720 --pyramid 3 --block_nums 3 --time_nums 2 --kernel 3 --learning_rate 0.0001 --dropout 0.1 --criterion Standard --test_inverse --d_model 32 --timebed None --itr 20 --reproducible

python -u main.py --model RT --data ETTh2 --features M  --label_len 168  --pred_len 24 --pyramid 3 --block_nums 3 --time_nums 2 --kernel 3 --learning_rate 0.0001 --dropout 0.1 --criterion Standard --test_inverse --d_model 448 --timebed None --group --angle 30 --itr 20 --reproducible

python -u main.py --model RT --data ETTh2 --features M  --label_len 168  --pred_len 48 --pyramid 3 --block_nums 3 --time_nums 2 --kernel 3 --learning_rate 0.0001 --dropout 0.1 --criterion Standard --test_inverse --d_model 448 --timebed None --group --angle 30 --itr 20 --reproducible

python -u main.py --model RT --data ETTh2 --features M  --label_len 168  --pred_len 168 --pyramid 3 --block_nums 3 --time_nums 2 --kernel 3 --learning_rate 0.0001 --dropout 0.1 --criterion Standard --test_inverse --d_model 448 --timebed None --group --angle 30 --itr 20 --reproducible

python -u main.py --model RT --data ETTh2 --features M  --label_len 168  --pred_len 336 --pyramid 3 --block_nums 3 --time_nums 2 --kernel 3 --learning_rate 0.0001 --dropout 0.1 --criterion Standard --test_inverse --d_model 448 --timebed None --group --angle 30 --itr 20 --reproducible

python -u main.py --model RT --data ETTh2 --features M  --label_len 168  --pred_len 720 --pyramid 3 --block_nums 3 --time_nums 2 --kernel 3 --learning_rate 0.0001 --dropout 0.1 --criterion Standard --test_inverse --d_model 112 --timebed None --group --angle 30 --itr 20 --reproducible

python -u main.py --model RT --data ETTm1 --features S  --label_len 336  --pred_len 24 --pyramid 3 --block_nums 3 --time_nums 2 --kernel 3 --learning_rate 0.0001 --dropout 0.1 --criterion Standard --test_inverse --d_model 32 --timebed None --batch_size 64 --itr 20 --reproducible

python -u main.py --model RT --data ETTm1 --features S  --label_len 336  --pred_len 48 --pyramid 3 --block_nums 3 --time_nums 2 --kernel 3 --learning_rate 0.0001 --dropout 0.1 --criterion Standard --test_inverse --d_model 32 --timebed None --batch_size 64 --itr 20 --reproducible

python -u main.py --model RT --data ETTm1 --features S  --label_len 336  --pred_len 96 --pyramid 3 --block_nums 3 --time_nums 2 --kernel 3 --learning_rate 0.0001 --dropout 0.1 --criterion Standard --test_inverse --d_model 32 --timebed None --batch_size 64 --itr 20 --reproducible

python -u main.py --model RT --data ETTm1 --features S  --label_len 336  --pred_len 288 --pyramid 3 --block_nums 3 --time_nums 2 --kernel 3 --learning_rate 0.0001 --dropout 0.1 --criterion Standard --test_inverse --d_model 32 --timebed None --batch_size 64 --itr 20 --reproducible

python -u main.py --model RT --data ETTm1 --features S  --label_len 336  --pred_len 672 --pyramid 3 --block_nums 3 --time_nums 2 --kernel 3 --learning_rate 0.0001 --dropout 0.1 --criterion Standard --test_inverse --d_model 32 --timebed None --batch_size 64 --itr 20 --reproducible

python -u main.py --model RT --data ETTm1 --features M  --label_len 336  --pred_len 24 --pyramid 3 --block_nums 3 --time_nums 2 --kernel 3 --learning_rate 0.0001 --dropout 0.1 --criterion Standard --test_inverse --d_model 112 --timebed None --group --angle 45 --itr 20 --reproducible

python -u main.py --model RT --data ETTm1 --features M  --label_len 336  --pred_len 48 --pyramid 3 --block_nums 3 --time_nums 2 --kernel 3 --learning_rate 0.0001 --dropout 0.1 --criterion Standard --test_inverse --d_model 112 --timebed None --group --angle 45 --itr 20 --reproducible

python -u main.py --model RT --data ETTm1 --features M  --label_len 336  --pred_len 96 --pyramid 3 --block_nums 3 --time_nums 2 --kernel 3 --learning_rate 0.0001 --dropout 0.1 --criterion Standard --test_inverse --d_model 112 --timebed None --group --angle 45 --itr 20 --reproducible

python -u main.py --model RT --data ETTm1 --features M  --label_len 336  --pred_len 288 --pyramid 3 --block_nums 3 --time_nums 2 --kernel 3 --learning_rate 0.0001 --dropout 0.1 --criterion Standard --test_inverse --d_model 112 --timebed None --group --angle 45 --itr 20 --reproducible

python -u main.py --model RT --data ETTm1 --features M  --label_len 336  --pred_len 672 --pyramid 3 --block_nums 3 --time_nums 2 --kernel 3 --learning_rate 0.0001 --dropout 0.1 --criterion Standard --test_inverse --d_model 112 --timebed None --group --angle 45 --itr 20 --reproducible

# Self-supervised
python -u main.py --model RT --data ETTh1 --features S  --label_len 168  --pred_list 24,48,168,336,720 --pyramid 3 --block_nums 3 --time_nums 2 --kernel 3 --learning_rate 0.0001 --dropout 0.1 --criterion Standard --test_inverse --d_model 32 --timebed None --batch_size 16  --cost_batch_size 32 --cost_epochs 10 --cost_grow_epochs 5 --forecasting_form Self-supervised --itr 20 --reproducible

python -u main.py --model RT --data ETTh2 --features S  --label_len 168  --pred_list 24,48,168,336,720 --pyramid 3 --block_nums 3 --time_nums 2 --kernel 3 --learning_rate 0.0001 --dropout 0.1 --criterion Standard --test_inverse --d_model 64 --timebed None --batch_size 32  --cost_batch_size 64 --cost_epochs 10 --cost_grow_epochs 5 --forecasting_form Self-supervised --itr 20 --reproducible

python -u main.py --model RT --data ETTm1 --features S  --label_len 336  --pred_list 24,48,96,288,672 --pyramid 3 --block_nums 3 --time_nums 2 --kernel 3 --learning_rate 0.0001 --dropout 0.1 --criterion Standard --test_inverse --d_model 32 --timebed None --batch_size 64  --cost_batch_size 128 --cost_epochs 10 --cost_grow_epochs 0 --forecasting_form Self-supervised --itr 20 --reproducible --aug_num 2 --block_shift 2

python -u main.py --model RT --data ETTh1 --features M  --label_len 168  --pred_list 24,48,168,336,720 --pyramid 3 --block_nums 3 --time_nums 2 --kernel 3 --learning_rate 0.0001 --dropout 0.1 --criterion Standard --test_inverse --d_model 112 --timebed None --batch_size 16  --cost_batch_size 32 --cost_epochs 10 --cost_grow_epochs 5 --group --angle 45 --forecasting_form Self-supervised --itr 20 --reproducible

python -u main.py --model RT --data ETTh2 --features M  --label_len 168  --pred_list 24,48,168,336,720 --pyramid 3 --block_nums 3 --time_nums 2 --kernel 3 --learning_rate 0.0001 --dropout 0.1 --criterion Standard --test_inverse --d_model 112 --timebed None --batch_size 32  --cost_batch_size 64 --cost_epochs 10 --cost_grow_epochs 5 --group --angle 30 --forecasting_form Self-supervised --itr 20 --reproducible

python -u main.py --model RT --data ETTm1 --features M  --label_len 336  --pred_list 24,48,96,288,672 --pyramid 3 --block_nums 3 --time_nums 2 --kernel 3 --learning_rate 0.0001 --dropout 0.1 --criterion Standard --test_inverse --d_model 112 --timebed None --batch_size 64  --cost_batch_size 128 --cost_epochs 10 --cost_grow_epochs 0 --group --angle 45 --forecasting_form Self-supervised --itr 20 --reproducible --aug_num 2 --block_shift 2


