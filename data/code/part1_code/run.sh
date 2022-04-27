
python /data/code/part1_code/train.py --h 360 --w 900 --fold 0 --checkpoints v4 --use_extra 0 --enhance_mode 3
python /data/code/part1_code/train.py --h 360 --w 900 --fold 1 --checkpoints v4 --use_extra 0 --enhance_mode 3
python /data/code/part1_code/train.py --h 360 --w 900 --fold 2 --checkpoints v4 --use_extra 0 --enhance_mode 3
python /data/code/part1_code/train.py --h 360 --w 900 --fold 3 --checkpoints v4 --use_extra 0 --enhance_mode 3
python /data/code/part1_code/train.py --h 360 --w 900 --fold 4 --checkpoints v4 --use_extra 0 --enhance_mode 3
python /data/code/part1_code/train.py --h 400 --w 1000 --fold 0 --checkpoints v6 --use_extra 1 --enhance_mode 2
python /data/code/part1_code/train.py --h 400 --w 1000 --fold 1 --checkpoints v6 --use_extra 1 --enhance_mode 2
python /data/code/part1_code/train.py --h 400 --w 1000 --fold 2 --checkpoints v6 --use_extra 1 --enhance_mode 2
python /data/code/part1_code/train.py --h 400 --w 1000 --fold 3 --checkpoints v6 --use_extra 1 --enhance_mode 2
python /data/code/part1_code/train.py --h 400 --w 1000 --fold 4 --checkpoints v6 --use_extra 1 --enhance_mode 2
python /data/code/part1_code/train_ema.py --checkpoints v7 --fold 0
python /data/code/part1_code/train_ema.py --checkpoints v7 --fold 1
python /data/code/part1_code/train_ema.py --checkpoints v7 --fold 2
python /data/code/part1_code/train_ema.py --checkpoints v7 --fold 3
python /data/code/part1_code/train_ema.py --checkpoints v7 --fold 4
python /data/code/part1_code/train.py --h 360 --w 900 --fold 0 --checkpoints vq1 --use_extra 1 --enhance_mode 3
python /data/code/part1_code/train.py --h 360 --w 900 --fold 1 --checkpoints vq1 --use_extra 1 --enhance_mode 3
python /data/code/part1_code/train.py --h 360 --w 900 --fold 2 --checkpoints vq1 --use_extra 1 --enhance_mode 3
python /data/code/part1_code/train.py --h 360 --w 900 --fold 3 --checkpoints vq1 --use_extra 1 --enhance_mode 3
python /data/code/part1_code/train.py --h 360 --w 900 --fold 4 --checkpoints vq1 --use_extra 1 --enhance_mode 3
python /data/code/part1_code/train_ema_epoch.py --checkpoints vq2 --fold 0
python /data/code/part1_code/train_ema_epoch.py --checkpoints vq2 --fold 1
python /data/code/part1_code/train_ema_epoch.py --checkpoints vq2 --fold 2
python /data/code/part1_code/train_ema_epoch.py --checkpoints vq2 --fold 3
python /data/code/part1_code/train_ema_epoch.py --checkpoints vq2 --fold 4
python /data/code/part1_code/train.py --h 400 --w 1000 --fold 0 --checkpoints vq3 --use_extra 1 --enhance_mode 1
python /data/code/part1_code/train.py --h 400 --w 1000 --fold 1 --checkpoints vq3 --use_extra 1 --enhance_mode 1
python /data/code/part1_code/train.py --h 400 --w 1000 --fold 2 --checkpoints vq3 --use_extra 1 --enhance_mode 1
python /data/code/part1_code/train.py --h 400 --w 1000 --fold 3 --checkpoints vq3 --use_extra 1 --enhance_mode 1
python /data/code/part1_code/train.py --h 400 --w 1000 --fold 4 --checkpoints vq3 --use_extra 1 --enhance_mode 1
python /data/code/part1_code/infer_oof.py --h 360 --w 900 --use_model v4
python /data/code/part1_code/infer_oof.py --h 400 --w 1000 --use_model v6
python /data/code/part1_code/infer_oof.py --h 400 --w 1000 --use_model v7
python /data/code/part1_code/infer_oof.py --h 360 --w 900 --use_model vq1
python /data/code/part1_code/infer_oof.py --h 360 --w 900 --use_model vq2
python /data/code/part1_code/infer_oof.py --h 400 --w 1000 --use_model vq3
python /data/code/part1_code/infer.py --h 360 --w 900 --use_model v4
python /data/code/part1_code/infer.py --h 400 --w 1000 --use_model v6
python /data/code/part1_code/infer.py --h 400 --w 1000 --use_model v7
python /data/code/part1_code/infer.py --h 360 --w 900 --use_model vq1
python /data/code/part1_code/infer.py --h 360 --w 900 --use_model vq2
python /data/code/part1_code/infer.py --h 400 --w 1000 --use_model vq3
python /data/code/part1_code/make_sub.py
