from torch.utils.data import DataLoader

from dataset.wiki_dataset import BERTDataset
from models.bert_model import *
from tqdm import tqdm

import torch
import pandas as pd
import numpy as np
import os

config = {}
config['train_corpus_path'] = '../corpus/train_wiki.txt'
config['test_corpus_path'] = '../corpus/test_wiki.txt'
config['word2idx_path'] = '../corpus/bert_word2idx_extend.json'
config['output_path'] = '../bert_state_dict'

config['batch_size'] = 1
config['max_seq_len'] = 250
config['vocab_size'] = 32162
config['lr'] = 5e-7
config['num_workers'] = 0
config['num_train_epochs'] = 5


class Pretrainer():
    def __init__(self, vocab_size, max_seq_len, batch_size, lr, with_cuda=True):
        self.vocab_size = vocab_size
        self.batch_size = batch_size
        self.lr = lr
        cuda_condition = torch.cuda.is_available() and with_cuda
        self.device = torch.device('cuda:0' if cuda_condition else 'cpu')
        self.max_seq_len = max_seq_len
        bertconfig = BertConfig(vocab_size=config['vocab_size'])
        self.bert_model = BertForPreTraining(config=bertconfig)
        self.bert_model.to(self.device)
        train_dataset = BERTDataset(config['train_corpus_path'],
                                    config['word2idx_path'],
                                    seq_len=self.max_seq_len,
                                    hidden_dim=bertconfig.hidden_size,
                                    on_memory=False)
        self.train_dataloader = DataLoader(train_dataset, batch_size=self.batch_size,
                                           num_workers=config['num_workers'],
                                           collate_fn=lambda x: x)

        test_dataset = BERTDataset(config['test_corpus_path'],
                                   config['word2idx_path'],
                                   seq_len=self.max_seq_len,
                                   hidden_dim=bertconfig.hidden_size,
                                   on_memory=True)
        self.test_dataloader = DataLoader(test_dataset, batch_size=self.batch_size,
                                          num_workers=config['num_workers'],
                                          collate_fn=lambda x: x)
        self.hidden_dim = bertconfig.hidden_size
        self.positional_enc = self.init_positional_encoding()
        self.positional_enc = torch.unsqueeze(self.positional_enc, dim=0)

        optim_parameters = list(self.bert_model.parameters())

        self.optimizer = torch.optim.Adam(optim_parameters, lr=self.lr)

        print('Total Parameters:', sum([p.nelement() for p in self.bert_model.parameters()]))

    def init_positional_encoding(self):
        '''
        初始化位置向量
        '''
        position_enc = np.array([
            [pos / np.power(10000, 2 * i / self.hidden_dim) for i in range(self.hidden_dim)]
            if pos != 0 else np.zeros(self.hidden_dim) for pos in range(self.max_seq_len)
        ])
        position_enc[1:, 0::2] = np.sin(position_enc[1:, 0::2])
        position_enc[1:, 1::2] = np.cos(position_enc[1:, 1::2])
        denominator = np.sqrt(np.sum(position_enc ** 2, axis=1, keepdims=True))
        position_enc = position_enc / (denominator + 1e-8)
        position_enc = torch.from_numpy(position_enc).type(torch.FloatTensor)
        return position_enc

    def load_model(self, model, dir_path='../output'):
        '''
        加载模型
        '''
        checkpoint_dir = self.find_most_recent_state_dict(dir_path)
        checkpoint = torch.load(checkpoint_dir)
        # todo
        model.load_state_dict(checkpoint['mode_state_dict'], strict=False)  # strict 是否要求字典key相同
        torch.cuda.empty_cache()
        model.to(self.device)
        print('{} loaded for training!'.format(checkpoint_dir))

    def train(self, epoch, df_path=config['output_path'] + '/df_log.pickle'):
        # 训练模式
        self.bert_model.train()
        self.iteration(epoch, self.train_dataloader, train=True, df_path=df_path)

    def test(self, epoch, df_path=config['output_path']+'/df_log.pickle'):
        # 设置eval模式
        self.bert_model.eval()
        # 不更新weight
        with torch.no_grad():
            return self.iteration(epoch, self.test_dataloader, train=False, df_path=df_path)


    def iteration(self, epoch, data_loader, train=True, df_path=config['output_path'] + '/df_log.pickle'):
        if not os.path.isfile(df_path):
            df = pd.DataFrame(columns=['epoch', 'train_next_sen_loss', 'train_mlm_loss',
                                       'train_next_sen_acc', 'train_mlm_acc',
                                       'test_next_sen_loss', 'test_mlm_loss',
                                       'test_next_sen_acc', 'test_mlm_acc'])
            df.to_pickle(df_path)
            print('log DataFrame created!')

        str_code = 'train' if train else 'test'

        # 设置进度条
        data_iter = tqdm(enumerate(data_loader),
                         desc='EP_%s:%d' % (str_code, epoch),
                         total=len(data_loader),
                         bar_format='{l_bar}{r_bar}')

        total_next_sen_loss = 0.0
        total_mlm_loss = 0.0
        total_next_sen_acc = 0.0
        total_mlm_acc = 0.0
        total_element = 0

        for i, data in data_iter:
            data = self.padding(data)
            # 0. batch_data will be sent into device
            data = {key: value.to(self.device) for key, value in data.items()}  # todo: key是什么
            # [batch_size, seq_len, embedding_dim]
            positional_enc = self.positional_enc[:, :data['bert_input'].size()[-1], :].to(self.device)

            # 1. forward the next_sentence_prediction and masked_lm_model
            mlm_preds, next_sen_preds = self.bert_model.forward(input_ids=data['bert_input'],
                                                                positional_enc=positional_enc,
                                                                token_type_ids=data['segment_label'])
            mlm_acc = self.get_mlm_accuracy(mlm_preds, data['bert_label'])
            next_sen_acc = next_sen_preds.argmax(dim=-1, keepdim=False).eq(data['is_next']).sum().item()
            mlm_loss = self.compute_loss(mlm_preds, data['bert_label'], self.vocab_size, ignore_index=0)
            next_sen_loss = self.compute_loss(next_sen_preds, data['is_next'])
            loss = mlm_loss + next_sen_loss

            # 3. backward and optimization only in train
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
                    'test_next_sen_loss': 0, 'test_mlm_loss': 0,
                    'test_next_sen_acc': 0, 'test_mlm_acc': 0
                }
            else:
                log_dict = {
                    'epoch': epoch,
                    'test_next_sen_loss': total_next_sen_acc / (i + 1),
                    'test_mlm_loss': total_mlm_loss / (i + 1),
                    'test_next_sen_acc': total_next_sen_acc / total_element,
                    'test_mlm_acc': total_mlm_acc / (i + 1),
                    'train_next_sen_loss': 0, 'train_mlm_loss': 0,
                    'train_next_sen_acc': 0, 'train_mlm_acc': 0
                }

            if i % 10 == 0:
                # todo
                data_iter.write(str({k: v for k, v in log_dict.items() if v != 0 and k != 'epoch'}))

        if train:
            # todo
            df = pd.read_pickle(df_path)
            df = df.append([log_dict])
            df.reset_index(inplace=True, drop=True) # inplace=True: 直接修改df, 不再创建新的对象；drop=True:
            df.to_pickle(df_path)
        else:
            log_dict = {k: v for k, v in log_dict.items() if v != 0 and k != 'epoch'}
            df = pd.read_pickle(df_path)
            df.reset_index(inplace=True, drop=True)
            for k, v in log_dict.items():
                df.at[epoch, k] = v
            df.to_pickle(df_path)
            return float(log_dict['test_next_sen_loss']) + float(log_dict['test_mlm_loss'])

    def padding(self, output_dic_list):
        # todo output_dic_list的具体格式
        bert_input = [i['bert_input'] for i in output_dic_list]
        bert_label = [i['bert_label'] for i in output_dic_list]
        segment_label = [i['segment_label'] for i in output_dic_list]
        bert_input = torch.nn.utils.rnn.pad_sequence(bert_input, batch_first=True)
        bert_label = torch.nn.utils.rnn.pad_sequence(bert_label, batch_first=True)
        segment_label = torch.nn.utils.rnn.pad_sequence(segment_label, batch_first=True)
        is_next = torch.cat([i['is_next'] for i in output_dic_list])
        return {
            'bert_input': bert_input,
            'bert_label': bert_label,
            'segment_label': segment_label,
            'is_next': is_next
        }

    def compute_loss(self, predictions, labels, num_class=2, ignore_index=None):
        # todo ignore_index 是什么，忽略#CLS#位？
        if ignore_index is None:
            loss_func = CrossEntropyLoss()
        else:
            loss_func = CrossEntropyLoss(ignore_index=ignore_index)
        return loss_func(predictions.view(-1, num_class), labels.view(-1))


    def get_mlm_accuracy(self, predictions, labels):
        predictions = torch.argmax(predictions, dim=-1, keepdim=True)
        # todo label=0 表示未被修改的token?
        mask = (labels > 0).to(self.device)
        # 只计算被MASK的token
        mlm_accuracy = torch.sum((predictions == labels) * mask).float()
        mlm_accuracy /= (torch.sum(mask).float() + 1e-8)
        return mlm_accuracy

    def find_most_recent_state_dict(self, dir_path):
        dic_list = [i for i in os.listdir(dir_path)]
        if len(dic_list) == 0:
            raise  FileNotFoundError('can not find any state dict in {}'.format(dir_path))
        dic_list = [i for i in dic_list if 'model' in i]
        dic_list = sorted(dic_list, key=lambda k: int(k.split('.')[-1]))
        return dir_path + '/' + dic_list[-1]

if __name__ == '__main__':

    def init_trainer(dynamic_lr, load_model=False):
        trainer = Pretrainer(
            vocab_size=config['vocab_size'],
            max_seq_len=config['max_seq_len'],
            batch_size=config['batch_size'],
            lr=dynamic_lr,
            with_cuda=True
        )

        if load_model:
            trainer.load_model(trainer.bert_model, dir_path=config['output_path'])
        return trainer

    start_epoch = 5
    train_epoches = 999
    trainer = init_trainer(config['lr'], load_model=True)

    all_loss = []
    threshold = 0
    patient = 10
    best_loss = 9999999999
    dynamic_lr = config['lr']

    # todo 为什么从star_epoch开始
    for epoch in range(start_epoch, start_epoch + train_epoches):
        print('train with learning rate {}'.format(str(dynamic_lr)))
        trainer.train(epoch)

        loss = trainer.test(epoch)

        all_loss.append(loss)
        best_loss = min(all_loss)
        if all_loss[-1] > best_loss:
            # 当前epoch是loss减小
            threshold += 1
            del trainer
            dynamic_lr *= 0.8
            trainer = init_trainer(dynamic_lr, load_model=True)
        else:
            # 当前epoch没有使loss减小
            del trainer
            dynamic_lr *= 1.05
            trainer = init_trainer(dynamic_lr, load_model=True)


        if threshold >= patient:
            print('epoch {} has the lowest loss.'.format(start_epoch + np.argmin(np.array(all_loss))))
            print('early stop!')
            break







