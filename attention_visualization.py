import torch
from models.bert_model import BertModel, BertConfig
from dataset.inference_dataloader import Preprocessing
import warnings
import json
import math
import os
import configparser
import matplotlib.pyplot as plt

from matplotlib.font_manager import FontProperties
font = FontProperties(fname='./SimHei.ttf')


class Plot_Attention:

    def __init__(self, max_seq_len, batch_size=1, with_cuda=True):
        # print(os.getcwd())

        # 加载配置文件
        parser = configparser.ConfigParser()
        parser.read("./config/model_config.ini")
        self.config = parser['DEFAULT']

        # 词量, 注意在这里实际字（词）汇量 = vocab_size - 20
        # 前20个token为特殊token, 比如 padding, cls
        self.vocab_size = int(self.config['vocab_size'])
        self.batch_size = batch_size

        # 是否使用GPU
        cuda_condition = torch.cuda.is_available() and with_cuda
        self.device = torch.device('cuda:0' if cuda_condition else 'cpu')

        # 限定最大句长
        self.max_seq_len = max_seq_len

        # 初始化超参数配置
        bertconfig = BertConfig(vocab_size=self.vocab_size)

        # 初始化Bert模型
        self.bert_mode = BertModel(config=bertconfig)
        self.bert_mode.to(self.device)

        # 加载字典
        self.word2idx = self.load_dic(self.config['word2idx_path'])

        # 初始化预处理器
        self.preprocess_batch = Preprocessing(hidden_dim=bertconfig.hidden_size,
                                              max_positions=max_seq_len,
                                              word2idx=self.word2idx)

        # 加载Bert预训练模型
        self.load_model(self.bert_mode, dir_path=self.config['state_dict_dir'])

        # 关闭dropout
        self.bert_mode.eval()

    def load_dic(self, path):
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def load_model(self, model, dir_path='./output'):
        '''加载模型'''
        checkpoint_dir = self.find_most_recent_state_dict(dir_path)
        checkpoint = torch.load(checkpoint_dir)

        # todo
        # 不加载masked language model 和 next sentence 的层和参数
        checkpoint['model_state_dict'] = {k[5:]: v for k, v in checkpoint['model_state_dict'].items()
                                          if k[:4] == 'bert'}

        model.load_state_dict(checkpoint['model_state_dict'], strict=True)

        torch.cuda.empty_cache()
        model.to(self.device)
        print("{} loaded for evaluation!".format(checkpoint_dir))

    def __call__(self, text, layer_num, head_num):
        # 获取attention矩阵的list
        attention_matrices = self.get_attention(text)

        # 准备热图的tags, 把字分开
        labels = [i + " " for i in list(text)]
        labels = ['#CLS#', ] + labels + ['#SEP#', ]

        # 画图
        plt.figure(figsize=(8, 8))
        plt.imshow(attention_matrices[layer_num][0][head_num])
        plt.yticks(range(len(labels)), labels, fontproperties=font, fontsize=18)
        plt.xticks(range(len(labels)), labels, fontproperties=font, fontsize=18)
        plt.show()

    def get_attention(self, text_list, batch_size=1):
        '''
        为了可视化, batch_size只能等于1
        '''
        if isinstance(text_list, str):
            text_list = [text_list, ]

        len_ = len(text_list)
        text_list = [i for i in text_list if len(i) != 0]

        if len(text_list) == 0:
            raise NotImplementedError('输入文本全部为空，长度为0！')
        if len(text_list) < len_:
            warnings.warn('输入的文本中有长度为0的句子，他们将被忽略掉！')

        max_seq_len = max(len(i) for i in text_list)

        # 预处理, 获取batch
        text_tokens, positional_enc = self.preprocess_batch(
            text_list, max_seq_len=max_seq_len
        )
        # 转换positional_enc的维度
        positional_enc = torch.unsqueeze(positional_enc, dim=0).to(self.device)

        attention_matrices = self.bert_mode.forward(input_ids=text_tokens,
                                                    positional_enc=positional_enc,
                                                    get_attention_matrices=True)

        # 因为batch_size=1 所以直接返回每层的注意力矩阵
        return [i.detach().numpy() for i in attention_matrices]


    def find_most_recent_state_dict(self, dir_path):
        # 找到模型存储的最新的state_dict路径
        dic_lis = [i for i in os.listdir(dir_path)]
        if len(dic_lis) == 0:
            raise FileNotFoundError("can not find any state dict in {}!".format(dir_path))
        dic_lis = [i for i in dic_lis if "model" in i]
        dic_lis = sorted(dic_lis, key=lambda k: int(k.split(".")[-1]))
        return dir_path + "/" + dic_lis[-1]

if __name__ == '__main__':

    model = Plot_Attention(max_seq_len=256)
    model('神经网络与深度学习', layer_num=3, head_num=2)
