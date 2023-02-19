# SoftVC VITS Singing Voice Conversion


## 模型简介
歌声音色转换模型，通过SoftVC内容编码器提取源音频语音特征，与F0同时输入VITS替换原本的文本输入达到歌声转换的效果。同时，更换声码器为 [NSF HiFiGAN](https://github.com/openvpi/DiffSinger/tree/refactor/modules/nsf_hifigan) 解决断音问题

### 4.0版本更新内容
+ 特征输入更换为 [Content Vec](https://github.com/auspicious3000/contentvec) 
+ 采样率统一使用44100hz
+ 由于更改了hop size等参数以及精简了部分模型结构，推理所需显存占用大幅降低，4.0版本44khz显存占用甚至远小于3.0版本的32khz
+ 调整了部分代码结构
+ 数据集制作、训练过程和3.0保持一致，但模型不通用
+ 增加了可选项 1：vc模式自动预测音高f0,即转换语音时不需要手动输入变调key，男女声的调能自动转换，但仅限语音转换，该模式转换歌声会跑调
+ 增加了可选项 2：通过kmeans聚类方案减小音色泄漏，即使得音色更加像目标音色

模型仍在训练测试效果中。。。。目前暂时请不要训练

## 预先下载的模型文件
+ contentvec ：[checkpoint_best_legacy_500.pt](https://ibm.box.com/s/z1wgl1stco8ffooyatzdwsqn2psd9lrr)
  + 放在`hubert`目录下
+ 预训练底模文件：还在训练中

## colab一键数据集制作、训练脚本
暂未制作

## 数据集准备
仅需要以以下文件结构将数据集放入dataset_raw目录即可
```shell
dataset_raw
├───speaker0
│   ├───xxx1-xxx1.wav
│   ├───...
│   └───Lxx-0xx8.wav
└───speaker1
    ├───xx2-0xxx2.wav
    ├───...
    └───xxx7-xxx007.wav
```


## 数据预处理
1. 重采样至 44100hz

```shell
python resample.py
 ```
2. 自动划分训练集 验证集 测试集 以及自动生成配置文件
```shell
python preprocess_flist_config.py
```
3. 生成hubert与f0
```shell
python preprocess_hubert_f0.py
```
执行完以上步骤后 dataset 目录便是预处理完成的数据，可以删除dataset_raw文件夹了


## 训练
```shell
python train.py -c configs/config.json -m 44k
```

## 推理
使用 [inference_main.py](inference_main.py)

截止此处，4.0使用方法（训练、推理）和3.0完全一致，没有任何变化（推理增加了命令行支持）

```shell
# 例
python inference_main.py -m "G_30400.pth" -c "configs/config.json" -n "君の知らない物語-src.wav" -t 0 -s "nen"
```
必填项部分
+ -m, --model_path：模型路径。
+ -c, --config_path：配置文件路径。
+ -n, --clean_names：wav 文件名列表，放在 raw 文件夹下。
+ -t, --trans：音高调整，支持正负（半音）。
+ -s, --spk_list：合成目标说话人名称。

可选项部分：见下一节
+ -a, --auto_predict_f0：语音转换自动预测音高，转换歌声时不要打开这个会严重跑调。
+ -cm, --cluster_model_path：聚类模型路径，如果没有训练聚类则随便填。
+ -cr, --cluster_infer_ratio：聚类方案占比，范围 0-1，若没有训练聚类模型则填 0 即可。

## 可选项
如果前面的效果已经满意，那以下内容可以忽略，不影响模型使用
### 自动f0预测
4.0模型训练过程会训练一个f0预测器，对于语音转换可以开启自动音高预测，如果效果不好也可以使用手动的，但转换歌声时请不要启用此功能！！！会严重跑调！！
+ 在inference_main中设置auto_predict_f0为true即可
### 聚类音色泄漏控制
介绍：聚类方案基本可以完全消除音色泄漏，使得模型训练出来更像目标的音色，但是单纯的聚类方案会降低模型的咬字（会口齿不清），本模型采用了融合的方式，
可以线性控制聚类方案与非聚类方案的占比，也就是可以手动在"像目标音色" 和 "咬字清晰" 之间调整比例，找到合适的折中点。
+ 训练过程：
  + 前面的已有步骤不用进行任何的变动，只需要额外训练一个聚类模型
  + 聚类模型的训练：使用cpu性能较好的机器训练，据我的经验在腾讯云6核cpu训练每个speaker需要约4分钟即可完成训练
  + 执行python cluster/train_cluster.py ，模型的输出会在 logs/44k/kmeans_10000.pt
+ 推理过程：
  + inference_main中指定cluster_model_path
  + inference_main中指定cluster_infer_ratio，0为完全不使用聚类，1为只使用聚类，通常设置0.5即可

## Onnx导出
暂未完成

