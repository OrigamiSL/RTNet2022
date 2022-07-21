# End-to-end
python -u main.py --model RT --data ECL --root_path ./data/ECL/ --features S --label_len 336  --pred_len 24 --pyramid 3 --block_nums 3 --time_nums 2 --kernel 3 --learning_rate 0.0001 --dropout 0.1 --criterion Standard --test_inverse --target MT_320 --d_model 128 --timebed None --batch_size 128 --itr 20 --reproducible

python -u main.py --model RT --data ECL --root_path ./data/ECL/ --features S --label_len 336  --pred_len 48 --pyramid 3 --block_nums 3 --time_nums 2 --kernel 3 --learning_rate 0.0001 --dropout 0.1 --criterion Standard --test_inverse --target MT_320 --d_model 128 --timebed None --batch_size 128 --itr 20 --reproducible

python -u main.py --model RT --data ECL --root_path ./data/ECL/ --features S --label_len 336  --pred_len 168 --pyramid 3 --block_nums 3 --time_nums 2 --kernel 3 --learning_rate 0.0001 --dropout 0.1 --criterion Standard --test_inverse --target MT_320 --d_model 128 --timebed None --batch_size 128 --itr 20 --reproducible

python -u main.py --model RT --data ECL --root_path ./data/ECL/ --features S --label_len 336  --pred_len 336 --pyramid 3 --block_nums 3 --time_nums 2 --kernel 3 --learning_rate 0.0001 --dropout 0.1 --criterion Standard --test_inverse --target MT_320 --d_model 128 --timebed None --batch_size 128 --itr 20 --reproducible

python -u main.py --model RT --data ECL --root_path ./data/ECL/ --features S --label_len 336  --pred_len 720 --pyramid 3 --block_nums 3 --time_nums 2 --kernel 3 --learning_rate 0.0001 --dropout 0.1 --criterion Standard --test_inverse --target MT_320 --d_model 128 --timebed None --batch_size 128 --itr 20 --reproducible

python -u main.py --model RT --data ECL --root_path ./data/ECL/ --features M --label_len 672  --Alter_label_len 168,336 --pred_len 24 --pyramid 3 --block_nums 3 --time_nums 2 --kernel 3 --learning_rate 0.0001 --dropout 0.1 --criterion Standard --test_inverse --target MT_320 --d_model 256 --timebed year --batch_size 128 --group --group_pred 80 --reproducible

python -u main.py --model RT --data ECL --root_path ./data/ECL/ --features M --label_len 672  --Alter_label_len 168,336 --pred_len 48 --pyramid 3 --block_nums 3 --time_nums 2 --kernel 3 --learning_rate 0.0001 --dropout 0.1 --criterion Standard --test_inverse --target MT_320 --d_model 256 --timebed year --batch_size 128 --group --group_pred 80 --reproducible

# Self-supervised
python -u main.py --model RT --data ECL --root_path ./data/ECL/ --features S --label_len 336  --pred_list 24,48,168,336,720 --pyramid 3 --block_nums 3 --time_nums 2 --kernel 3 --learning_rate 0.0001 --dropout 0.1 --criterion Standard --test_inverse --target MT_320 --d_model 64 --timebed None --batch_size 16  --cost_batch_size 64 --cost_epochs 10 --cost_grow_epochs 5 --forecasting_form Self-supervised --itr 20 --reproducible

python -u main.py --model RT --data ECL --root_path ./data/ECL/ --features M --label_len 672  --Alter_label_len 168,336 --pred_list 24,48 --pyramid 3 --block_nums 3 --time_nums 2 --kernel 3 --learning_rate 0.0001 --dropout 0.1 --criterion Standard --test_inverse --target MT_320 --d_model 64 --timebed None --batch_size 16 --cost_batch_size 64 --cost_epochs 10 --forecasting_form Self-supervised --group --group_pred 321 --reproducible