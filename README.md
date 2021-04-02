<h1 align="center">
<p>TensorflowTTS</p>
<p align="center">
<img alt="python" src="https://img.shields.io/badge/python-%3E%3D3.6-blue">
<img alt="tensorflow" src="https://img.shields.io/badge/tensorflow-%3E%3D2.2.0-orange">
</p>
</h1>
<h2 align="center">
<p>集成了Tensorflow 2版本的端到端语音合成模型，并且RTF(实时率)在0.1左右</p>
</h2>
<p align="center">
目前集成了中文的Tacotron2/FastSpeech 两种结构
</p>
<p align="center">
当前还在开发阶段，暂时没有文本前端处理
</p>
<p align="center">
欢迎使用并反馈bug

# 其它项目

ASR：https://github.com/Z-yq/TensorflowASR

NLU:  -

BOT:  -

## Pretrained Model

预训练模型为中间模型，未充分训练。

**RTF**(实时率) 测试于**CPU**单核合成任务。 


Model Name| link                                          |code       |Params Size|RTF|
----------|-----------------------------------------------|-----------|---------|-----|
Tacotron2|https://pan.baidu.com/s/1g2qvifBUtpyIMp7E2N19Xw|wfwv| 34M|0.114|
FastSpeech|https://pan.baidu.com/s/18WphipZTW-aNY82ISlU8Nw|tjru| 28M|0.056|
Melgan|https://pan.baidu.com/s/13QSRG6Rb1N0vaJFB2MYiGg|r6e3|24M|0.02|


**快速使用：**

下载预训练模型，修改 tacotron.yml/fastspeech.yml 里的目录参数（outdir），并在修改后的目录中添加 checkpoints 目录，

将model_xx.h5(xx为数字)文件放入对应的checkpoints目录中，

修改run-test.py中的读取的config文件（tacotron.yml/fastspeech.yml）路径，运行run-test.py即可。


## Supported Structure
-  **Tacotron2**
-  **FastSpeech**
-  **Melgan**


## Requirements

-   Python 3.6+
-   Tensorflow 2.2+: `pip install tensorflow`
-   librosa
-   pypinyin `if you need use the default phoneme`
-   addons `pip install tensorflow-addons`
-   tqdm
-   pesq

## Usage

1. 准备train_list.


    **声学特征模型** 格式，其中'\t'为tap:
    
    ```text
    file_path1 \t text1 \t spkid
    file_path2 \t text2 \t spkid
    ……
    ```
    
    **声码器** 格式:
    ```text
    file_path1
    file_path2
    ……
    ```


​        
2.修改配置文件 **common.yml** 和模型配置文件 **tacotron.yml/vocoder.yml**来自定义自己的模型。
3.然后执行命令:

    ```shell
    python train_acoustic.py --data_config ./configs/common.yml --model_config ./configs/tacotron.yml
    python train_vocoder.py --data_config ./configs/common.yml --model_config ./configs/vocoder.yml
    ```

4.想要测试时，可以参考 **_`run-test.py`_** 里写的demo.


5.执行脚本`tacotron_extract_features.py`
   ```shell
    python tacotron_extract_features.py --data_config ./configs/common.yml --model_config ./configs/tacotron.yml
   ```
6.finetune训练的vocoder，适配tacotron生成的mel图。设置**vocoder.yml**中的 `load_from_npz` 为 `True`,以及`adjust_type` 设置为`tacotron`
   ```shell 
    python train_vocoder.py --data_config ./configs/common.yml --model_config ./configs/vocoder.yml
   ```

**以下为使用fastspeech模型流程，如果不使用fastspeech可忽略。**

7.训练fastspeech。执行脚本：
   ```shell
    python train_acoustic.py --data_config ./configs/common.yml --model_config ./configs/fastspeech.yml
   ```
8.执行脚本`fastspeech_extract_features.py`
   ```shell
    python fastspeech_extract_features.py --data_config ./configs/common.yml --model_config ./configs/fastspeech.yml
   ```
9.finetune训练的vocoder，适配fastspeech生成的mel图。设置**vocoder.yml**中的 `load_from_npz` 为 `True`,以及`adjust_type` 设置为`fastspeech`
   ```shell 
    python train_vocoder.py --data_config ./configs/common.yml --model_config ./configs/vocoder.yml
   ```
以上即完成整个流程

## Tips
如果你想用你自己的音素，需要对应 `utils/text_featurizers.py` 里的`extract`方法。

不要忘记你的音素列表用 **_`/S`_** 打头,e.g:


        /S
        d
        sh
        ……


## References

感谢关注：


https://github.com/TensorSpeech/TensorFlowTTS `modify from it`


## Licence

允许并感谢您使用本项目进行学术研究、商业产品生产等，但禁止将本项目作为商品进行交易。

Overall, Almost models here are licensed under the Apache 2.0 for all countries in the world.

Allow and thank you for using this project for academic research, commercial product production, allowing unrestricted commercial and non-commercial use alike. 

However, it is prohibited to trade this project as a commodity.
