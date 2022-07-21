# End-to-end
python -u main.py --model RT --data WTH --root_path ./data/WTH/ --features S --label_len 48  --pred_len 24 --pyramid 4 --block_nums 4 --time_nums 2 --kernel 3 --learning_rate 0.0001 --dropout 0 --criterion Standard --test_inverse --target WetBulbCelsius --d_model 32 --timebed year --itr 20 --reproducible

python -u main.py --model RT --data WTH --root_path ./data/WTH/ --features S --label_len 48  --pred_len 48 --pyramid 4 --block_nums 4 --time_nums 2 --kernel 3 --learning_rate 0.0001 --dropout 0 --criterion Standard --test_inverse --target WetBulbCelsius --d_model 32 --timebed year --itr 20 --reproducible

python -u main.py --model RT --data WTH --root_path ./data/WTH/ --features S --label_len 48  --pred_len 168 --pyramid 4 --block_nums 4 --time_nums 2 --kernel 3 --learning_rate 0.0001 --dropout 0 --criterion Standard --test_inverse --target WetBulbCelsius --d_model 32 --timebed year --itr 20 --reproducible

python -u main.py --model RT --data WTH --root_path ./data/WTH/ --features S --label_len 48  --pred_len 336 --pyramid 4 --block_nums 4 --time_nums 2 --kernel 3 --learning_rate 0.0001 --dropout 0 --criterion Standard --test_inverse --target WetBulbCelsius --d_model 32 --timebed year --itr 20 --reproducible

python -u main.py --model RT --data WTH --root_path ./data/WTH/ --features S --label_len 48  --pred_len 720 --pyramid 4 --block_nums 4 --time_nums 2 --kernel 3 --learning_rate 0.0001 --dropout 0 --criterion Standard --test_inverse --target WetBulbCelsius --d_model 32 --timebed year --itr 20 --reproducible

python -u main.py --model RT --data WTH --root_path ./data/WTH/ --features M --label_len 48  --pred_len 24 --pyramid 4 --block_nums 4 --time_nums 2 --kernel 3 --learning_rate 0.0001 --dropout 0 --criterion Standard --test_inverse --target WetBulbCelsius --d_model 64 --timebed year --itr 20 --reproducible

python -u main.py --model RT --data WTH --root_path ./data/WTH/ --features M --label_len 48  --pred_len 48 --pyramid 4 --block_nums 4 --time_nums 2 --kernel 3 --learning_rate 0.0001 --dropout 0 --criterion Standard --test_inverse --target WetBulbCelsius --d_model 64 --timebed year --itr 20 --reproducible

python -u main.py --model RT --data WTH --root_path ./data/WTH/ --features M --label_len 48  --pred_len 168 --pyramid 4 --block_nums 4 --time_nums 2 --kernel 3 --learning_rate 0.0001 --dropout 0 --criterion Standard --test_inverse --target WetBulbCelsius --d_model 64 --timebed year --itr 20 --reproducible

python -u main.py --model RT --data WTH --root_path ./data/WTH/ --features M --label_len 48  --pred_len 336 --pyramid 4 --block_nums 4 --time_nums 2 --kernel 3 --learning_rate 0.0001 --dropout 0 --criterion Standard --test_inverse --target WetBulbCelsius --d_model 64 --timebed year --itr 20 --reproducible

python -u main.py --model RT --data WTH --root_path ./data/WTH/ --features M --label_len 48  --pred_len 720 --pyramid 4 --block_nums 4 --time_nums 2 --kernel 3 --learning_rate 0.0001 --dropout 0 --criterion Standard --test_inverse --target WetBulbCelsius --d_model 64 --timebed year --itr 20 --reproducible

# Self-supervised
python -u main.py --model RT --data WTH --root_path ./data/WTH/ --features S --label_len 48  --pred_list 24,48,168,336,720 --pyramid 4 --block_nums 4 --time_nums 2 --kernel 3 --learning_rate 0.0001 --dropout 0 --criterion Standard --test_inverse --target WetBulbCelsius --d_model 32 --timebed year --batch_size 32  --cost_batch_size 64 --cost_epochs 10 --cost_grow_epochs 5 --forecasting_form Self-supervised --itr 20 --reproducible

python -u main.py --model RT --data WTH --root_path ./data/WTH/ --features M --label_len 48  --pred_list 24,48,168,336,720 --pyramid 4 --block_nums 4 --time_nums 2 --kernel 3 --learning_rate 0.0001 --dropout 0 --criterion Standard --test_inverse --target WetBulbCelsius --d_model 64 --timebed year --batch_size 32  --cost_batch_size 64 --cost_epochs 10 --cost_grow_epochs 5 --forecasting_form Self-supervised --itr 20 --reproducible

