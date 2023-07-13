# 多模态情感分析

对于给定配对的文本和图像，设计了一个多模态融合模型，预测对应的情感标签。

## Setup

> 请下载bert-base-uncased模型于bert-base-uncased文件夹中（huggingface加载文件时通常会出现反爬的报错）
> 配置依赖环境

- pandas==2.0.3

- numpy==1.24.2

- transformers==4.30.2

- torch==2.0.0+cu118

- scikit_learn==1.3.0

- Pillow==10.0.0

- Pillow==9.4.0

- chardet==4.0.0

```python
pip install -r requirements.txt
```
## Repository structure

lab文件主要结构

```python 
|-- lab5
    |-- main.py # model、config等主要函数
    |-- data_process.py # text和label，尤其是text格式处理
    |-- README.md
    |-- requirements.txt
    |-- data # 存储了所有数据集
    |   |-- test.json
    |   |-- test_without_label.txt
    |   |-- train.json
    |   |-- train.txt
    |   |-- data # image&text
    |-- bert-base-uncased # bert模型存储位置
    |-- output # 存储了保存的模型和预测文件
```

## Run Code

- 默认 同时进行train和test（双模态img+text）,不使用已经训练好的模型【如已有训练好的模型末尾加上--test_only --use_pretrained实现加载模型仅测试】
  在末尾加上--image_model="vgg"(或者resnet)调用对应image模型，默认为vgg模型
```python
python main.py 
```

- 做消融实验
  - 仅使用text 
    ```python
    python main.py --text_only
    ```
  - 仅使用image
    ```python
    python main.py --img_only
    ```

## Attribution

参考资料：
- [bert使用手册](https://www.jianshu.com/p/4e139a3260fd)
- [pytorch迁移学习_vgg](https://blog.csdn.net/weixin_42632271/article/details/107683469)
- [torchnision库学习](https://zhuanlan.zhihu.com/p/476220305)
- [debug](https://zhuanlan.zhihu.com/p/443146137)
- Very Deep Convolutional Networks for Large-Scale Image Recognition (vgg模型)