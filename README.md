# SoftVC VITS Singing Voice Conversion
## 模型简介
歌声音色转换模型，通过SoftVC内容编码器提取源音频语音特征，结合F0输入VITS替换原本的文本输入达到歌声转换的效果。

## 实现细节
+ F0转换为coarse F0 embedding后与soft-units相加替代vits原本的文本输入
+ 使用NSF-HiFiGAN 替代了VITS自带的HiFiGAN 感谢[zhaohui8969](https://github.com/zhaohui8969)
