from torch.utils.data import DataLoader
from dataset.wiki_dataset import BERTDataset
from models.bert_model import *

from tqdm import tqdm

import numpy as np
import pandas as pd
import os

config = {}
config['train_corpus_path'] = './corpus/train_wiki.txt'
config['test_corpus_path'] = './corpus/test_wiki.txt'
config['word2idx_path'] = './corpus/bert_word2idx_extend.json'
config['output_path'] = './output_wiki_bert'

config['batch_size'] = 1
config['max_seq_len'] = 200
config['vocab_size'] = 32162
config['lr'] = 2e-6
config['num_workers'] = 0

class Pretrainer:
    def __init__(self, bert_model,
                 vocab_size, max_seq_len,
                 batch_size, lr, with_cuda=True):
        # 词量, 注意这里实际字(词)汇量 = vocab_size - 20
        # 因为前20个token用来做一些特殊功能，如padding等
        self.vocab_size = vocab_size
        self.batch_size = batch_size
        self.lr = lr
        cuda_condition = torch.cuda.is_available() and with_cuda
        self.device = torch.device('cuda:0' if cuda_condition else 'cpu')
        # 限定单句最大长度
        self.max_seq_len = max_seq_len
        # 初始化超参数的配置
        bertconfig = BertConfig(vocab_size=config['vocab_size'])
        # 初始化bert模型
        self.bert_model = bert_model(config=bertconfig)
        self.bert_model.to(self.device)
        # 初始化训练数据集
        train_dataset = BERTDataset(corpus_path=config['train_corpus_path'],
                                    word2idx_path=config['word2idx_path'],
                                    seq_len=self.max_seq_len,
                                    hidden_dim=bertconfig.hidden_size,
                                    on_memory=False)
        # 初始化训练dataloader
        self.train_dataloader = DataLoader(train_dataset,
                                           batch_size=config['batch_size'],
                                           num_workers=config['num_workers'],
                                           collate_fn=lambda x:x)
        # 初始化测试数据集
        test_dataset = BERTDataset(corpus_path=config['test_corpus_path'],
                                   word2idx_path=config['word2idx_path'],
                                   seq_len=self.max_seq_len,
                                   hidden_dim=bertconfig.hidden_size,
                                   on_memory=True)
        # 初始化测试dataloader
        self.test_dataloader = DataLoader(test_dataset, batch_size=self.batch_size,
                                          num_workers=config['num_workers'],
                                          collate_fn=lambda x: x)
        # 初始化positional_encoding [max_seq_len, hidden_size]
        self.positional_enc = self.init_positional_encoding(hidden_dim=bertconfig.hidden_size,
                                                            max_seq_len=self.max_seq_len)
        # 拓展positional_encoding的维度为[1, max_seq_len, hidden_size]
        self.positional_enc = torch.unsqueeze(self.positional_enc, dim=0)

        # 列举需要优化的参数并传入优化器
        optim_parameters = list(self.bert_model.parameters())
        self.optimizer = torch.optim.Adam(optim_parameters, lr=self.lr)

        print('Total Parameters:', sum(p.nelement() for p in self.bert_model.parameters()))


    def init_positional_encoding(self, hidden_dim, max_seq_len):
        position_enc = np.array([
            [pos / np.power(10000, 2 * i / hidden_dim) for i in range(hidden_dim)]
            if pos != 0 else np.zeros(hidden_dim) for pos in range(max_seq_len)
        ])

        # dim=2i
        position_enc[1:, 0::2] = np.sin(position_enc[1:, 0::2])
        # dim=2i+1
        position_enc[1:, 1::2] = np.sin(position_enc[1:, 1::2])
        # todo 归一化处理 why? 用位置嵌入的每一行除以它的模长
        denominator = np.sqrt(np.sum(position_enc**2, axis=1, keepdims=True)) # 作为分母
        position_enc /= (denominator + 1e-8)
        position_enc = torch.from_numpy(position_enc).type(torch.FloatTensor)
        return position_enc

    def test(self, epoch, df_path='./output_wiki_bert/df_log.pickle'):
        self.bert_model.eval()
        with torch.no_grad():
            return self.iteration(epoch, self.test_dataloader, train=False, df_path=df_path)

    def load_model(self, model, dir_path='./output'):
        # 加载模型
        checkpoint_dir = self.find_most_recent_state_dict(dir_path)
        checkpoint = torch.load(checkpoint_dir)
        # todo key在哪保存的
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        torch.cuda.empty_cache()
        model.to(self.device)
        print('{} loaded for training!'.format(checkpoint_dir))


    def train(self, epoch, df_path='./output_wiki_bert/df_log.pickle'):
        self.bert_model.train()
        self.iteration(epoch, self.train_dataloader, train=True, df_path=df_path)

    def compute_loss(self, preditions, labels, num_class=2, ignore_index=None):
        if ignore_index is None:
            loss_func = CrossEntropyLoss()
        else:
            loss_func = CrossEntropyLoss(ignore_index=ignore_index)
        return loss_func(preditions.view(-1, num_class), labels.view(-1))


    def get_mlm_accuracy(self, predictions, labels):
        # predictions [batch_size, seq_len, vocab_size]
        predictions = torch.argmax(predictions, dim=-1, keepdim=False)  # predictions: [batch_size, seq_len]
        # labels: [batch_size, seq_len]
        mask = (labels > 0)  # 只考虑被MASK的token
        # 预测正确的数量
        pred_correct = torch.sum((predictions == labels) * mask).float()
        # accuracy
        mlm_accuracy = pred_correct / (torch.sum(mask).float() + 1e-8)
        return mlm_accuracy.item()

    def padding(self, output_dic_list):
        # todo output_dic_list的格式
        # [batch_size, seq_len, embed_dim]
        bert_input = [i['bert_input'] for i in output_dic_list]
        bert_label = [i['bert_label'] for i in output_dic_list]
        segment_label = [i['segment_label'] for  i in output_dic_list]
        # padding
        bert_input = torch.nn.utils.rnn.pad_sequence(bert_input, batch_first=True)
        bert_label = torch.nn.utils.rnn.pad_sequence(bert_label, batch_first=True)
        segment_label = torch.nn.utils.rnn.pad_sequence(segment_label, batch_first=True)
        # [batch_size]
        is_next = torch.cat([i['is_next'] for i in output_dic_list])
        return {
            'bert_input': bert_input,
            'bert_label': bert_label,
            'segment_label': segment_label,
            'is_next': is_next
        }

    def find_most_recent_state_dict(self, dir_path):
        if not os.path.exists(dir_path):
            os.mkdir(dir_path)
        dic_list = [i for i in os.listdir(dir_path)]
        if len(dic_list) == 0:
            raise FileNotFoundError('can not find any state dict in {}'.format(dir_path))
        # todo model什么时候存放的？
        dic_list = [i for i in dic_list if 'model' in i]
        dic_list = sorted(dic_list, key=lambda k: int(k.split('.')[-1]))
        return dir_path + '/' + dic_list[-1]

    def iteration(self, epoch, data_loader, train=True, df_path='./output_wiki_bert/df_log.pickle'):
        if not os.path.isfile(df_path) and epoch != 0:
            raise RuntimeError("log DataFrame path not found and can't create a new one because we're not training from scratch!")

        if not os.path.isfile(df_path) and epoch == 0:
            df = pd.DataFrame(columns=['epoch', 'train_next_sen_loss', 'train_mlm_loss',
                                       'train_next_sen_acc', 'train_mlm_acc',
                                       'test_next_sen_loss', 'test_mlm_loss',
                                       'test_next_sen_acc', 'test_mlm_acc'])
            df.to_pickle(df_path)
            print('log DataFrame created!')

        str_code = 'train' if train else 'test'

        # 设置进度条，得到迭代器对象
        data_iter = tqdm(enumerate(data_loader),
                         desc='EP_%s:%d' % (str_code, epoch),
                         total=len(data_loader),
                         bar_format='{l_bar}{r_bar}')

        total_next_sen_loss = 0
        total_mlm_loss = 0
        total_next_sen_acc = 0
        total_mlm_acc = 0
        total_element = 0

        for i, data in data_iter:
            data = self.padding(data)
            # 0. batch_data will be sent into the device
            data = {key: value.to(self.device) for key, value in data.items()}
            # todo data['bert_input'] 的维度
            positional_enc = self.positional_enc[:, :data['bert_input'].size()[-1], :].to(self.device)

            # 1. forward the next_sentence_prediction and masked_lm_model
            # mlm_preds: [batch_size, seq_len, vocab_size]
            # next_sen_preds: [batch_size, seq_len]
            mlm_preds, next_sen_preds = self.bert_model.forward(input_ids=data['bert_input'],
                                                                positional_enc=positional_enc,
                                                                token_type_ids=data['segment_label'])

            mlm_acc = self.get_mlm_accuracy(mlm_preds, data['bert_label'])
            next_sen_acc = next_sen_preds.argmax(dim=-1, keepdim=False).eq(data['is_next']).sum().item()
            mlm_loss = self.compute_loss(mlm_preds, data['bert_label'], self.vocab_size, ignore_index=0)
            next_sen_loss = self.compute_loss(next_sen_preds, data['is_next'])
            # 两个任务联合训练
            loss = mlm_loss + next_sen_loss

            # 3. 反向传播和梯度更新
            if train:
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            total_next_sen_loss += next_sen_loss.item()
            total_mlm_loss += mlm_loss.item()
            total_next_sen_acc += next_sen_acc
            total_element += data['is_next'].nelement()
            total_mlm_acc += mlm_acc

            if train:
                log_dict = {
                    'epoch': epoch,
                    'train_next_sen_loss': total_next_sen_loss / (i + 1),
                    'train_mlm_loss': total_mlm_loss / (i + 1),
                    'train_next_sen_acc': total_next_sen_acc / total_element,
                    'train_mlm_acc': total_mlm_acc / (i + 1),
                    'test_next_sen_loss': 0, 'test_mlm_loss':0,
                    'test_next_sen_acc':0, 'test_mlm_acc':0
                }
            else:
                log_dict = {
                    'epoch': epoch,
                    'test_next_sen_loss': total_next_sen_loss / (i + 1),
                    'test_mlm_loss': total_mlm_loss / (i + 1),
                    'test_next_sen_acc': total_next_sen_acc / total_element,
                    'test_mlm_acc': total_mlm_acc / (i + 1),
                    'train_next_sen_loss': 0, 'train_mlm_loss': 0,
                    'train_next_sen_acc': 0, 'train_mlm_acc': 0
                }

            if i % 10 == 0:
                data_iter.write(str({k: v for k, v in log_dict.items() if v != 0 and k != 'epoch'}))

        if train:
            df = pd.read_pickle(df_path)
            # 将日志信息追加到df中
            df = df.append([log_dict])
            # 重置索引
            df.reset_index(inplace=True, drop=True)
            # 保存到本地
            df.to_pickle(df_path)
        else:
            log_dict = {k: v for k, v in log_dict.items() if v != 0 and k != 'epoch'}
            df = pd.read_pickle(df_path)
            df.reset_index(inplace=True, drop=True)
            for k, v in log_dict.items():
                df.at[epoch, k] = v
            df.to_pickle(df_path)
            return float(log_dict['test_next_sen_loss']) + float(log_dict['test_mlm_loss'])


    def save_state_dict(self, model, epoch, dir_path='./output', file_path='bert.model'):
        if not os.path.exists(dir_path):
            os.mkdir(dir_path)

        save_path = dir_path + '/' + file_path + '.epoch.{}'.format(str(epoch))
        model.to('cpu')
        torch.save({'model_state_dict': model.state_dict()}, save_path)
        print('{} saved!'.format(save_path))
        model.to(self.device)


if __name__ == '__main__':

    def init_trainer(dynamic_lr, load_model=False):
        trainer = Pretrainer(BertForPreTraining,
                             vocab_size=config['vocab_size'],
                             max_seq_len=config['max_seq_len'],
                             batch_size=config['batch_size'],
                             lr=dynamic_lr,
                             with_cuda=True)

        if load_model:
            trainer.load_model(trainer.bert_model, dir_path=config['output_path'])
        return trainer


    start_epoch = 3
    train_epoches = 1
    trainer = init_trainer(config['lr'], load_model=True)

    all_loss = []
    threshold = 0
    patient = 10
    best_f1 = 0
    dynamic_lr = config['lr']

    # todo start_epoch 为什么要从3开始
    for epoch in range(start_epoch, start_epoch + train_epoches):
        print('train with learning rate {}'.format(str(dynamic_lr)))
        trainer.train(epoch)

        trainer.save_state_dict(trainer.bert_model, epoch, dir_path=config['output_path'],
                                file_path='bert.model')
        trainer.test(epoch)














