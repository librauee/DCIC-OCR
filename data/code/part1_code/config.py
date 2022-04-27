# 图像大小

fold = 0
IMAGE_HEIGHT = 256
IMAGE_WIDTH = 256
# 图像分类的类别
num_classes = 248
# 训练batchsize大小
batch_size = 32
# 训练epoch
num_epoch = 100
# 学习率
lr = 1e-3
# 训练过程保存模型地址
checkpoints = 'checkpoints'

# 训练集和验证集
train_dir = '/data/raw_data/train'
val_dir = '/data/user_data/val'
test_dir = '/data/raw_data/test'
