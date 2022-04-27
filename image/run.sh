## 安装依赖
pip install -r /data/user_data/requirements.txt

python /data/code/gen_moredata.py 

## 生成模型1
python /data/code/part2_code/train.py 
python /data/code/part2_code/inference.py 

## 生成模型2
python /data/code/part1_code/get_folds.py 
sh /data/code/part1_code/run.sh 
