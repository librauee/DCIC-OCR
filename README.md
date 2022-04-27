# 基于文本字符的交易验证码识别
* A榜第3、B榜第6

## 解决方案

* 使用基于多分类的思路，以effb4为主要模型
* 使用多种数据增强手段进行增强，包括mixup、cutmix、fmix等
* 使用imagecaptcha库进行数据生成
* 使用EMA增强模型性能
* 使用贝叶斯优化提升模型融合效果


## 运行步骤
1. 下载image文件夹内的镜像并用7z解压
2. 将训练、测试数据分别放置于raw_data/train,raw_data/test
3. 运行image/run.sh脚本即可得到结果