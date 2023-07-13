import argparse
import os  #解决了之前路径欠考虑的问题！
import warnings

import numpy as np
import json
import chardet
from PIL import Image
import pandas as pd

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader,Dataset

from transformers import AutoModel,AutoTokenizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report, accuracy_score
from torchvision import transforms
from torchvision.models import resnet50, vgg19_bn 
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class config():
    root_path = os.getcwd()
    data_dir = os.path.join(root_path, './data/data')
    train_data_path = os.path.join(root_path, './data/train.json') # 把原文件转换成了json格式
    test_data_path = os.path.join(root_path, './data/test.json')
    output_test_path = os.path.join(root_path, './output/test.txt')
    output_model_path = os.path.join(root_path, './output/model.pth') # 记得可以直接加载模型的位置，changepth名字哦！！！
 
    use_pretrained  = False
    num_labels = 3
    epoch = 20
    learning_rate = 5e-5 # 5e-5 1e-5 3e-3
    weight_decay = 1e-2 #1e-3
    loss_weight = [1, 5.78 , 2.03]
    # 由于positive比重过大，neutral比重过小，我们在loss中添加了weight函数以减轻样本的不平衡的问题
    '''
    有两种设置方法：
    1、样本数的倒数作为权重
    2、max(number of occurrences in most common class) / (number of occurrences in rare classes)
        即使用类别中最大样本数量除以当前类别样本的数量作为权重系数
        train中 positive 2424 negative 1194 neutral 419
        我们决定系数为 1 2424/1194=2.03 2424/419=5.78
    '''
    only = None
    
    middle_hidden_size = 64
    attention_nhead = 8
    attention_dropout = 0.4
    fuse_dropout = 0.5
    out_hidden_size = 128
    
    text_name = './bert-base-uncased' # 注意路径修改（模型已经被下载下来）！！！！
    text_learning_rate = 5e-6 # 3e-3
    text_dropout = 0.2
    
    image_name = 'resnet'
    image_size = 224 # 与创建的空白图片size一致
    image_learning_rate = 5e-6 # 3e-3
    image_dropout = 0.2
    img_hidden_seq = 64

# 读取数据，返回[(guid, text, img, label)]元组列表 【将复杂的数据初处理拎出来，此时相对简易，所有原函数的测试都是在ipynb中试验后写入】
def read_from_file(path, data_dir, only=None): # json、data、onlyorboth
    data = []
    json_file = pd.read_json(path).values.tolist()
    for d in json_file:
        guid, label, text = str(d[0]), d[1], d[2]
        if only == 'text': 
            img = Image.new(mode='RGB', size=(224, 224), color=(0, 0, 0)) #
        else:
            img_path = os.path.join(data_dir, guid + '.jpg')
            img = Image.open(img_path)
            img.load()
        if only == 'img': 
            text = ''
        data.append((guid, text, img, label))
    return data

# based 框架 __init__ __len__ __getitem__
class Vocab():  # 一个简单的label - id 映射vocab
    def __init__(self,label=['positive','neutral','negative','null']) :
        self.label2id = {token: idx for idx, token in enumerate(label)}
        self.id2label = {idx: token for token, idx in self.label2id.items()}
    def __len__(self):
        return len(self.label2id)
    def __getitem__(self, tokens_or_indices): # label2id_ str int ;id2label_ int str
        # 我们需要让Vocab支持正反向查找和序列索引 
        # isinstance(object, classinfo)  object -- 实例对象 classinfo -- 可以是直接或间接类名、基本类型或者由它们组成的元组。
        # 单个索引情形
        if isinstance(tokens_or_indices, (str, int)):
            # 找不到指定的键值时返回未知词元（索引）
            return self.label2id.get(tokens_or_indices, 2) if isinstance(tokens_or_indices, str) else self.id2label.get(tokens_or_indices, 'null ')
        # 多个索引情形
        elif isinstance(tokens_or_indices, (list, tuple)):
            return [self.__getitem__(item) for item in tokens_or_indices]
        else:
            raise TypeError

# 如果只在dataloader处写入collate_fn函数【尝试】
class MyDataset(Dataset):
    def __init__(self, guids, texts, imgs, labels) -> None:
        self.guids = guids
        self.texts = texts
        self.imgs = imgs
        self.labels = labels
    def __len__(self):
        return len(self.guids)
    def __getitem__(self, index):
        return self.guids[index], self.texts[index], self.imgs[index], self.labels[index]

    # 手动将抽取出的样本堆叠起来的函数
    def collate_fn(self, batch): 
        guids = [b[0] for b in batch]
        texts = [torch.LongTensor(b[1]) for b in batch]
        imgs = torch.FloatTensor([np.array(b[2]).tolist() for b in batch]) 
        labels = torch.LongTensor([b[3] for b in batch])
         
        texts_mask = [torch.ones_like(text) for text in texts] # 处理到同一长度
        paded_texts = pad_sequence(texts, batch_first=True, padding_value=0) # pad
        paded_texts_mask = pad_sequence(texts_mask, batch_first=True, padding_value=0).gt(0) 
        # pad 标记为0

        return guids, paded_texts, paded_texts_mask, imgs, labels
    
class Processor():
    def __init__(self, config):
        self.config = config
        self.labelvocab = Vocab()
        pass
    def forward(self, data, batch_size, shuffle):

        tokenizer = AutoTokenizer.from_pretrained(self.config.text_name)
         # image处理_
        '''
        Image.open()
        Image.transform(size, method, data, resample, fill) # 输出尺寸（处理原始图像）、转换方法、 对转换方法的额外数据、 可选的重采样过滤器
        transforms.CenterCrop 图片中心进行裁剪
        transforms.ColorJitter 图像颜色的对比度、饱和度和零度进行变黄
        transforms.FiveCrop 对图像四个角和中心进行裁剪得到五分图像
        transforms.Grayscale  对图像进行灰度变换
        transforms.Pad  使用固定值进行像素填充
        transforms.RandomAffine  随机仿射变换 
        transforms.RandomCrop  随机区域裁剪
        transforms.RandomHorizontalFlip  随机水平翻转
        transforms.RandomRotation  随机旋转
        transforms.RandomVerticalFlip  随机垂直翻转
        '''
        img_transform = transforms.Compose([
                    transforms.Resize(self.config.image_size),
                    transforms.CenterCrop(self.config.image_size),
                    transforms.RandomHorizontalFlip(0.5),
                    transforms.ToTensor(), # 转换为tensor
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # pytorch教程
        ])

        guids, encoded_texts, encoded_imgs, encoded_labels = [], [], [], []
        for line in data:
            guid, text, img, label = line
            guids.append(guid)
            text.replace('#', '')
            # [CLS] 标志放在第一个句子的首位，经过BERT 得到的的表征向量C 可以用于后续的分类任务。 [SEP] 标志用于分开两个输入句子，例如输入句子A 和B，要在句子A，B 后面增加[SEP] 标志。
            tokens = tokenizer.tokenize('[CLS]' + text + '[SEP]') 
            encoded_texts.append(tokenizer.convert_tokens_to_ids(tokens))
            encoded_imgs.append(img_transform(img))
            encoded_labels.append(self.labelvocab[label]) # vocab没有加载成功！！！
        
        dataset_inputs = guids, encoded_texts, encoded_imgs, encoded_labels
        dataset = MyDataset(*dataset_inputs)
        return DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=dataset.collate_fn)

class Trainer():

    def __init__(self, config, processor, model):
        self.config = config
        self.processor = processor
        self.model = model.to(device)       
        # Model
        text_params = set(self.model.text_model.text.parameters())
        image_params = set(self.model.img_model.full_image.parameters())
        other_params = list(set(self.model.parameters()) - text_params - image_params)
        no_decay = ['bias', 'LayerNorm.weight']
        # params = model.parameters()
        '''
        named_parameters() 方法_可以对一个nn.Module中所有注册的参数进行迭代
        named_children()方法_来查看这个nn.Module的直接子级的模块
        named_modules()方法，这个方法会循环遍历nn.Module以及其child nn.Modules ,其实与named_children()的主要区别就是遍历的程度是否更deeper
        '''
        # 不衰退! 默认衰退通过参数定义
        params = [
            {'params': [p for n, p in self.model.text_model.text.named_parameters() if not any(nd in n for nd in no_decay)],
                'lr': self.config.text_learning_rate, 'weight_decay': self.config.weight_decay},
            {'params': [p for n, p in self.model.text_model.text.named_parameters() if any(nd in n for nd in no_decay)],
                'lr': self.config.text_learning_rate, 'weight_decay': 0.0},
            {'params': [p for n, p in self.model.img_model.full_image.named_parameters() if not any(nd in n for nd in no_decay)],
                'lr': self.config.image_learning_rate, 'weight_decay': self.config.weight_decay},
            {'params': [p for n, p in self.model.img_model.full_image.named_parameters() if any(nd in n for nd in no_decay)],
                'lr': self.config.image_learning_rate, 'weight_decay': 0.0},
            {'params': other_params,
                'lr': self.config.learning_rate, 'weight_decay': self.config.weight_decay},
        ]   # reference
        
        # self.optimizer = torch.optim.Adam(params, lr=config.learning_rate)
        self.optimizer = torch.optim.AdamW(params, lr=config.learning_rate) # 加入了weight_decay

    def train(self, train_loader):
        self.model.train() # 调整为train模式
        loss_list = []
        true_labels, pred_labels = [], []
        for id, batch in enumerate(train_loader):
            guids, texts, texts_mask, imgs, labels = batch # collate_fn
            texts, texts_mask, imgs, labels = texts.to(device), texts_mask.to(device), imgs.to(device), labels.to(device)
            pred, loss = self.model(texts, texts_mask, imgs, labels=labels)
            loss_list.append(loss.item())
            true_labels.extend(labels.tolist())
            pred_labels.extend(pred.tolist())
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        
        train_loss = round(sum(loss_list) / len(loss_list), 5)
        return train_loss, loss_list  
    
    # 模型中labels=None 默认只有在传入labels计算损失，为训练和验证
    def valid(self, val_loader):
        self.model.eval()
        val_loss = 0
        true_labels, pred_labels = [], []
        for batch in val_loader:
            guids, texts, texts_mask, imgs, labels = batch
            texts, texts_mask, imgs, labels = texts.to(device), texts_mask.to(device), imgs.to(device), labels.to(device)
            pred, loss = self.model(texts, texts_mask, imgs, labels=labels)
            val_loss += loss.item()
            true_labels.extend(labels.tolist())
            pred_labels.extend(pred.tolist())    
        metrics = accuracy_score(true_labels, pred_labels)
        # print(classification_report(true_labels, pred_labels))
        return val_loss / len(val_loader), metrics        
    def predict(self, test_loader):
        self.model.eval()
        pred_guids, pred_labels = [], []
        for batch in test_loader:
            guids, texts, texts_mask, imgs, labels = batch
            texts, texts_mask, imgs = texts.to(device), texts_mask.to(device), imgs.to(device)
            pred = self.model(texts, texts_mask, imgs)
            pred_guids.extend(guids)
            pred_labels.extend(pred.tolist())
        return [(guid, label) for guid, label in zip(pred_guids, pred_labels)]

class TextModel(nn.Module):

    def __init__(self, config):
        super(TextModel, self).__init__()

        self.text = AutoModel.from_pretrained(config.text_name)
        self.trans = nn.Sequential(
            nn.Dropout(config.text_dropout),
            nn.Linear(self.text.config.hidden_size, config.middle_hidden_size),
            nn.ReLU(inplace=True)
        ) 
        
        # 微调：阶段预训练模型网络的softmax层（last）替换掉；使用较小的学习率进行模型训练， ！ 冻结预训练网络前几层的权重亦可【数据量小】
        # 可以多进行两次尝试（未知相关性，       
        for param in self.text.parameters():
            param.requires_grad = True # 默认不进行微调

    def forward(self, bert_inputs, masks, token_type_ids=None):
        # 请保证 bert_inputs.shape == masks.shape
        # assert bert_inputs.shape == masks.shape, 'error! bert_inputs and masks must have same shape!'
        bert_out = self.text(input_ids=bert_inputs, token_type_ids=token_type_ids, attention_mask=masks)
        pooler_out = bert_out['pooler_output']
        
        return self.trans(pooler_out)
    
class ImageResNetModel(nn.Module):

    def __init__(self, config):
        super(ImageResNetModel, self).__init__()
        self.full_image = resnet50(pretrained=True)
        
        # 迁移学习  
        # print(list(self.image_.children()))
        self.resnet = nn.Sequential(
            *(list(self.full_image.children())[:-1]),
            nn.Flatten() # 一般写在某个神经网络模型之后，用于对神经网络模型的输出进行处理，得到tensor类型的数据
            # 从第一维度展平到最后一个维度
        )

        self.trans = nn.Sequential(
            nn.Dropout(config.image_dropout),
            nn.Linear(self.full_image.fc.in_features,config.middle_hidden_size),
            nn.ReLU(inplace=True)
        ) #与text——保持一致，以确保最后两个特征可以融合进行预测
        
        # 默认不进行微调，可以调整试试
        for param in self.full_image.parameters():
            param.requires_grad = False

    def forward(self, imgs):
        feature = self.resnet(imgs)
        # batchsize, in_features
        return self.trans(feature)  

class ImageVGGModel(nn.Module):

    def __init__(self, config):
        super(ImageVGGModel, self).__init__()
        self.full_image = vgg19_bn(pretrained=True)
        '''
        VGG(
            (features): #此
            (avgpool):
            (classifier):
        )  
        ''' 
        
        classifier = nn.Sequential(
            *(list(self.full_image.children())[-1])[:6]
        )

        self.trans = nn.Sequential(
            nn.Dropout(config.image_dropout),
            nn.Linear(list(self.full_image.classifier.children())[-1].in_features, config.middle_hidden_size),
            nn.ReLU(inplace=True)
        )
        
        self.full_image.classifier = classifier
        # 默认不进行微调，可以调整试试
        for param in self.full_image.parameters():
            param.requires_grad = False

    def forward(self, imgs):
        feature = self.full_image(imgs)
        # batchsize, in_features
        return self.trans(feature)
    


    
class Model(nn.Module):

    def __init__(self, config):
        super(Model, self).__init__()
        
        self.text_model = TextModel(config)
       
        self.img_model = ImageVGGModel(config)  if(config.image_name=="vgg") else ImageResNetModel(config)

        # 分类器可以进行一些修改
        self.text_classifier = nn.Sequential(
            nn.Dropout(config.fuse_dropout),
            nn.Linear(config.middle_hidden_size, config.out_hidden_size),
            nn.ReLU(inplace=True),
            nn.Dropout(config.fuse_dropout),
            nn.Linear(config.out_hidden_size, config.num_labels),
            nn.Softmax(dim=1)
        )
        self.img_classifier = nn.Sequential(
            nn.Dropout(config.fuse_dropout),
            nn.Linear(config.middle_hidden_size, config.out_hidden_size),
            nn.ReLU(inplace=True),
            nn.Dropout(config.fuse_dropout),
            nn.Linear(config.out_hidden_size, config.num_labels),
            nn.Softmax(dim=1)
        )
        self.loss_func = nn.CrossEntropyLoss(weight=torch.tensor(config.loss_weight))

    def forward(self, texts, texts_mask, imgs, labels=None):
        text_feature = self.text_model(texts, texts_mask)
        img_feature = self.img_model(imgs)

        text_prob_vec = self.text_classifier(text_feature)
        img_prob_vec = self.img_classifier(img_feature)
        prob_vec = torch.softmax((text_prob_vec + img_prob_vec), dim=1)
        pred_labels = torch.argmax(prob_vec, dim=1)

        if labels is not None:
            loss = self.loss_func(prob_vec, labels)
            return pred_labels, loss
        else:
            return pred_labels
        
def train(processor):
    data = read_from_file(config.train_data_path, config.data_dir, config.only)
    train_data, val_data = train_test_split(data, train_size=0.8, test_size=0.2, random_state=22)

    train_loader = processor.forward(train_data, 16, True)
    val_loader = processor.forward(val_data, 16, False)

    best_acc = trainer.valid(val_loader) if config.use_pretrained else 0
    print("start training...")
    epoch = config.epoch
    for i in range(epoch):
        loss, _ = trainer.train(train_loader)
        print(f'[Epoch {i + 1}]  loss: {loss:.4f}')
        
        loss, vac = trainer.valid(val_loader)
        print('Valid Loss: {}'.format(loss), 'Valid Acc: {}'.format(vac))

        if vac > best_acc:
            best_acc = vac
            torch.save(model.state_dict(), config.output_model_path)
        print()

def test(processor):
    test_data = read_from_file(config.test_data_path, config.data_dir, config.only)
    test_loader = processor.forward(test_data, 8, False)

    outputs = trainer.predict(test_loader)
    formated_outputs = ['guid,tag']
    for guid, label in outputs:
        formated_outputs.append((str(guid) + ',' + processor.labelvocab[label]))

    return formated_outputs

if __name__ == "__main__":
    # 终端指令
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_only', action='store_true', help='训练_测试 or 测试only')
    parser.add_argument('--text_only', action='store_true', help='文本only')
    parser.add_argument('--img_only', action='store_true', help='图像only')
    parser.add_argument('--use_pretrained', default='0', type=int, help='是否使用保存的模型')
    parser.add_argument('--image_model',type=str,default="vgg",help="图像处理模型")
    args = parser.parse_args()

    # 需要满足相斥条件，此时为both
    config.only = 'img' if args.img_only else None
    config.only = 'text' if args.text_only else None
    if args.img_only and args.text_only: 
        config.only = None 
    # only我们将用空白代替！ read_from_file 中会处理
    
    config.use_pretrained = True if args.use_pretrained==1 else False # 默认情况不需要使用
    config.img_name = args.image_model
    
    # 结构from deeplearning
    processor = Processor(config)
    model = Model(config)
    trainer = Trainer(config, processor, model) 
 
    if(config.use_pretrained):
        model.load_state_dict(torch.load(config.load_model_path))

    if args.test_only:
        pass
    else:
        train(processor)
    
    predict = test(processor)
    with open(config.output_test_path, 'w') as f:
        for line in predict:
            f.write(line)
            f.write('\n')
        f.close()
    