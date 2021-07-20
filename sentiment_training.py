from torch.utils.data import DataLoader

from dataset.sentiment_dataset import CLSDataset
from models.bert_sentiment_analysis import *
from sklearn import metrics
from metrics import *
from tqdm import tqdm

import numpy as np
import pandas as pd
import configparser
import os
import json

class Sentiment_trainer():

    def __init__(self, max_seq_len, batch_size, lr, with_cuda=True):
        config_ = configparser.ConfigParser()
        config_.read('./config/sentiment_model_config.ini.ini')
        self.config = config_['DEFAULT']
        self.vocab_size = int(self.config['vocab_size'])
        self.batch_size = batch_size
        self.lr = lr
        # 加载字典
        with open(self.config['word2idx_path'], 'r', encoding='utf-8') as f:
            self.word2idx = json.load(f)

        cude_condition = torch.cuda.is_available() and with_cuda
        self.device = torch.device('cuda:0' if cude_condition else 'cpu')

        # 允许的序列最大长度
        self.max_seq_len = max_seq_len
        # 定义模型超参数
        bertconfig = BertConfig(vocab_size=self.vocab_size)
        # 初始化bert_sentiment_analysis模型
        self.bert_sentiment_model = Bert_Sentiment_Analysis(config=bertconfig)
        # 将模型发送到计算设备
        self.bert_sentiment_model.to(self.device)
        # 声明训练数据集
        train_dataset = CLSDataset(corpus_path=self.config['train_corpus_path'],
                                   word2idx=self.word2idx,
                                   max_seq_len=self.max_seq_len,
                                   data_regularization=True)
        self.train_dataloader = DataLoader(train_dataset, batch_size=self.batch_size,
                                           num_workers=0, collate_fn=lambda x: x) # 为了动态padding
        # 声明测试数据集
        test_dataset = CLSDataset(corpus_path=self.config['test_corpus_path'],
                                  word2idx=self.word2idx,
                                  max_seq_len=self.max_seq_len,
                                  data_regularization=False)
        self.test_dataloader = DataLoader(test_dataset, batch_size=self.batch_size,
                                          num_workers=0, collate_fn=lambda x: x)

        # 初始位置编码
        self.positional_enc = self.init_positional_encoding()
        self.positional_enc = torch.unsqueeze(self.positional_enc, dim=0) # [batch_size, seq_len, embed_dim]
        self.hidden_dim = bertconfig.hidden_size
        # 优化器
        self.optim_parameters = list(self.bert_sentiment_model.parameters())

        # learning rate
        self.init_optimizer(lr=self.lr)
        if not os.path.exists(self.config['state_dict_dir']):
            os.mkdir(self.config['state_dict_dir'])

    def init_optimizer(self, lr):
        '''使用Adam优化器'''
        # weight_decay: 相当于L2正则
        self.optimizer = torch.optim.Adam(self.optim_parameters, lr=lr, weight_decay=1e-3)

    def init_positional_encoding(self):
        '''初始化位置encoding'''
        position_enc = np.array([
            [pos / np.power(10000, 2 * i / self.hidden_dim) for i in range(self.hidden_dim)]
            if pos !=0 else np.zeros(self.hidden_dim) for pos in range(self.max_seq_len)
        ])

        position_enc[1:, 0::2] = np.sin(position_enc[1:, 0::2])
        position_enc[1:, 1::2] = np.sin(position_enc[1:, 1::2])
        # 归一化
        denominator = np.sqrt(np.sum(position_enc**2, axis=1, keepdims=True))
        position_enc = position_enc / (denominator + 1e-8)
        position_enc = torch.from_numpy(position_enc).type(torch.FloatTensor)
        return position_enc

    def load_model(self, model, dir_path='../output', load_bert=False):
        checkpoint_dir = self.find_recent_state_dict(dir_path)
        checkpoint = torch.load(checkpoint_dir)
        # 情感分析模型刚开始训练的时候，需要载入预训练的Bert
        # 这里我们不载入模型原本用于训练Next Sentence的pooler，而是重新初始化了一个
        # todo 模型载入
        if load_bert:
            checkpoint['model_state_dict'] = {k[5:]: v for k, v in checkpoint['model_state_dict'].items()
                                              if k[:4] == 'bert' and 'pooler' not in k}

        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        torch.cuda.empty_cache()
        model.to(self.device)
        print('{} loaded!'.format(checkpoint_dir))

    def train(self, epoch):
        self.bert_sentiment_model.train()
        self.iteration(epoch, self.train_dataloader, train=True)

    def test(self, epoch):
        '''
        一个epoch的测试，返回测试集的AUC
        :return: AUC
        '''
        self.bert_sentiment_model.eval()
        with torch.no_grad():
            return self.iteration(epoch, self.test_dataloader, train=False)

    def padding(self, output_dic_list):
        text_input = [i['text_input'] for i in output_dic_list]
        text_input = torch.nn.utils.rnn.pad_sequence(text_input, batch_first=True)
        label = torch.cat([i['label'] for i in output_dic_list])
        return {
            'text_input': text_input,
            'label': label
        }

    def find_recent_state_dict(self, dir_path):
        dic_list = [i for i in os.listdir(dir_path)]
        if len(dic_list) == 0:
            raise FileNotFoundError('can not find any state dict in {}'.format(dir_path))
        dic_list = [i for i in  dic_list if 'model' in i]
        dic_list = sorted(dic_list, key=lambda k: int(k.split('.')[-1]))
        return dir_path + '/' + dic_list[-1]


    def save_state_dict(self, model, epoch, state_dict_dir='../output', file_path='bert.model'):
        '''存储当前模型参数'''
        if not os.path.exists(state_dict_dir):
            os.mkdir(state_dict_dir)
        save_path = state_dict_dir + '/' + file_path + '.epoch.{}'.format(str(epoch))
        model.to('cpu')
        torch.save({'model_state_dict': model.state_dict()}, save_path)
        print('{} saved!'.format(save_path))
        model.to(self.device)


    def iteration(self, epoch, data_loader, train=True, df_name='df_log.pickle'):
        df_path = self.config['state_dict_dir'] + '/' + df_name
        if not os.path.isfile(df_path):
            df = pd.DataFrame(columns=['epoch', 'train_loss', 'train_auc', 'test_loss', 'test_auc'])
            df.to_pickle(df_path)
            print('log DataFrame created!')

        str_code = 'train' if train else 'test'

        # 进度条显示
        data_iter = tqdm(enumerate(data_loader),
                         desc='EP_%s:%d' % (str_code, epoch),
                         total=len(data_loader),
                         bar_format='{l_bar}{r_bar}')

        total_loss = 0
        all_preds, all_labels = [], []

        for i, data in data_iter:
            data = self.padding(data)
            # 将数据发送到计算设备上
            data = {key: value.to(self.device) for key, value in data.items()}
            # 截取positional_enc并发送到计算设备上
            positional_enc = self.positional_enc[:, data['text_input'].size()[-1], :].to(self.device)

            # 正向传播，得到预测结果和loss
            preds, loss = self.bert_sentiment_model.forward(text_input=data['text_input'],
                                                            positional_enc=positional_enc,
                                                            labels=data['label'])
            # todo preds [batch_size]
            preds = preds.detach().cpu().numpy().reshape(-1).tolist()
            labels = data['label'].cpu().numpy().reshape(-1).tolist()
            all_preds.extend(preds)
            all_labels.extend(labels)

            # 计算AUC
            fpr, tpr, thresholds = metrics.roc_curve(y_true=all_labels,
                                                     y_score=all_preds)
            auc = metrics.auc(fpr, tpr)

            # todo 先计算AUC 再找最优的threshold是不是不合理

            # 方向传播
            if train:
                # 清空之前的gradient
                self.optimizer.zero_grad()
                # backward
                loss.backward()
                # update weights
                self.optimizer.step()

            total_loss += loss.item()

            if train:
                log_dict = {
                    'epoch': epoch,
                    'train_loss': total_loss / (i+1),
                    'train_auc': auc,
                    'test_loss': 0, 'test_auc':0
                }
            else:
                log_dict = {
                    'epoch': epoch,
                    'train_loss': 0,
                    'train_auc': 0,
                    'test_loss': total_loss / (i+1), 'test_auc': auc
                }

            if i % 10 == 0:
                data_iter.write(str({k: v for k, v in log_dict.items() if v != 0}))

        threshold_ = find_best_threshold(all_predictions=all_preds, all_labels=all_labels)
        print(str_code + ' best threshold:' + str(threshold_))

        # 将当前epoch的情况记录到DataFrame中
        if train:
            df = pd.read_pickle(df_path)
            df = df.append([log_dict])
            df.reset_index(inplace=True, drop=True)
            df.to_pickle(df_path)
        else:
            log_dict = {k: v for k, v in log_dict.items() if v != 0 and k != 'epoch'}
            df = pd.read_pickle(df_path)
            df.reset_index(inplace=True, drop=True)
            for k, v in log_dict.items():
                df.at[epoch, k] = v
            df.to_pickle(df_path)
            return auc

if __name__ == '__main__':
    def init_trainer(dynamic_lr, batch_size=24):
        trainer = Sentiment_trainer(max_seq_len=300,
                                    batch_size=batch_size,
                                    lr = dynamic_lr,
                                    with_cuda=True)
        return trainer, dynamic_lr

    start_epoch = 0
    train_epoches = 9999
    trainer, dynamic_lr = init_trainer(dynamic_lr=1e-6, batch_size=24)

    all_auc = []
    threshold = 999
    patient = 10
    best_loss = 9999999999

    for epoch in range(start_epoch, start_epoch + train_epoches):
        if epoch == start_epoch and epoch == 0:
            '''第一个epoch的训练需要加载预训练的BERT模型'''
            trainer.load_model(trainer.bert_sentiment_model, dir_path='./bert_state_dict', load_bert=True)
        elif epoch == start_epoch:
            trainer.load_model(trainer.bert_sentiment_model, dir_path=trainer.config['state_dict_dir'])
        print('train with learning rate {}'.format(str(dynamic_lr)))
        # 训练一个epoch
        trainer.train(epoch)
        # 保存当前epoch模型参数
        trainer.save_state_dict(trainer.bert_sentiment_model, epoch,
                                state_dict_dir=trainer.config['state_dict_dir'],
                                file_path='sentiment.model')

        auc = trainer.test(epoch)

        all_auc.append(auc)
        best_auc = max(all_auc)

        if all_auc[-1] < best_auc:
            threshold += 1
            dynamic_lr *= 0.8
            trainer.init_optimizer(lr=dynamic_lr)
        else:
            threshold = 0

        if threshold >= patient:
            print('epoch {} has the lowest loss.'.format(start_epoch + np.argmax(np.array(all_auc))))
            print('early stop!')
            break




