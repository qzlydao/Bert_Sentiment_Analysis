import copy
import math
import torch

from torch import nn
from torch.nn import functional as F
from torch.nn import CrossEntropyLoss

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

def gelu(x):
    '''
    gelu(Gaussian Error Linear Unit)
    gelu(x) = xP(X<x), 可以用tanh或sigmoid函数近似替代
    gelu(x) = 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
    gelu(x) = x * sigmoid(1.702 * x)
    '''
    return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))

ACT2FN = {'gelu': gelu, 'relu': F.relu}

class BertConfig(object):
    '''
    Configuration class to store the configuration of a BertModel
    '''
    def __init__(self,
                 vocab_size, # 字典数
                 hidden_size=384,
                 num_hidden_layers=6, # transformer block的个数
                 num_attention_heads=12, # 多头个数
                 intermediate_size=384*4, # feedforward层线性映射的维度
                 hidden_act='gelu',
                 hidden_dropout_prob=0.4,
                 attention_probs_dropout_prob=0.4,
                 max_position_embeddings=512*2,
                 type_vocab_size=256, # 用来做next_sentence预测，这里预留256个分类，其实我们只用得到0，1
                 initializer_range=0.02 # 用来初始化模型参数的标准差
                 ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.initializer_range = initializer_range


class BertEmbeddings(nn.Module):
    '''
    LayerNorm层
    Construct the embeddings from word, position and token_type embeddings.
    '''
    def __init__(self, config):
        super(BertEmbeddings, self).__init__()
        # vocab_size * hidden_size
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=0)
        # type_vocab_size * hidden_size
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)
        # embedding矩阵初始化
        nn.init.orthogonal_(self.word_embeddings.weight) # 正交初始化
        nn.init.orthogonal_(self.token_type_embeddings.weight) # 正交初始化

        # embedding矩阵进行归一化
        # TODO 为什么要归一化
        epsilon = 1e-8
        self.word_embeddings.weight.data = self.word_embeddings.weight.data.div(
            # p=2 2范数
            torch.norm(self.word_embeddings.weight, p=2, dim=1, keepdim=True).data + epsilon
        )
        self.token_type_embeddings.weight.data = self.token_type_embeddings.weight.data.div(
            torch.norm(self.token_type_embeddings.weight, p=2, dim=1, keepdim=True).data + epsilon
        )

        self.LayerNorm = BertLayerNorm(config.hidden_size, epsilon=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_idxs, positional_enc, token_type_ids=None):
        '''
        :param input_idx:       [batch_size, seq_length]
        :param positional_enc:  [seq_length, embedding_dim]
        :param token_type_ids:  bert训练的时候，第一句是0，第二句是1
        :return: [batch_size, seq_len, embedding_dim]
        '''
        # 字向量查表
        word_embeddings = self.word_embeddings(input_idxs) # [batch_size, seq_length, hidden_dim]

        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_idxs) # [batch_size, seq_length]
        token_type_embeddings = self.token_type_embeddings(token_type_ids) # [batch_size, seq_length, hidden_dim]

        # TODO 为什么要加上token_type_embeddings?
        embeddings = word_embeddings + positional_enc + token_type_embeddings

        # 经过LayerNorm层和dropout层
        embeddings = self.LayerNorm(embeddings)

        # TODO dropout是作用在输入上的，而不是weight上？
        embeddings = self.dropout(embeddings)

        return embeddings



class BertLayerNorm(nn.Module):
    '''
    LayerNorm层
    '''
    def __init__(self, hidden_size, epsilon=1e-12):
        super(BertLayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = epsilon

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        # 对应位置元素相乘
        return self.weight * x + self.bias

class BertSelfAttention(nn.Module):
    '''
    自注意力机制
    '''
    def __init__(self, config):
        super(BertSelfAttention, self).__init__()
        # 判断embedding dim是否可以被num_attention_heads整除
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                'The hidden size (%d) is not a multiple of the number of attention heads (%d)' % (config.hidden_size, config.num_attention_heads)
            )

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        # K, Q, V 线性映射
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        # x为QKV中的一个，维度[batch_size, seq_length, embedding_dim]
        # 输出的维度经过reshape和转置：[batch_size, num_heads, seq_length, embedding_dim / num_heads]
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size) # [batch_size, seq_length, num_heads, attention_head_size]
        x = x.view(*new_x_shape) # [batch_size, seq_length, num_heads, attention_head_size]
        return x.permute(0, 2, 1, 3) # [batch_size, num_heads, seq_length, embedding_dim / num_heads]

    def forward(self, hidden_states, attention_mask, get_attention_matrices=False):
        # QKV线性变换，维度[batch_size, seq_length, embedding_dim]
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        # 将KQV分割成num_heads分
        # 把维度转换为 [batch_size, num_heads, seq_len, embedding_dim / num_heads]
        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Q与K求点积，计算attention_score [batch_size, num_heads, seq_len, seq_len]
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        # 除以K的dimension，开平方根以归一化
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # 加上attention_mask（使batch中padding位置上值区域无穷大）
        attention_scores = attention_scores + attention_mask # attention_mask: [batch_size, 1, 1, seq_len]

        # softmax归一化后，得到注意力矩阵
        attention_probs_ = nn.Softmax(dim=-1)(attention_scores)

        # dropout
        attention_probs = self.dropout(attention_probs_)

        # 用注意力矩阵加权V
        context_layer = torch.matmul(attention_probs, value_layer) # [batch_size, num_heads, seq_len, embedding_dim / num_heads]
        # 把加权后的V reshape, 得到[batch_size, seq_len, embedding_dim]
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size, )
        context_layer = context_layer.view(*new_context_layer_shape)

        # 输出attention矩阵用来可视化
        if get_attention_matrices:
            return context_layer, attention_probs_
        return context_layer, None


class BertSelfOutput(nn.Module):
    # 封装的LayerNorm和残差连接，用于处理self-attention的输出
    def __init__(self, config):
        super(BertSelfOutput, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = BertLayerNorm(config.hidden_size, epsilon=1e-12)
        self.droput = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.droput(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)

        return hidden_states


class BertAttention(nn.Module):
    # 封装的多头注意力机制部分，包括LayerNorm和残差连接
    def __init__(self, config):
        super(BertAttention, self).__init__()
        self.self = BertSelfAttention(config)
        self.output = BertSelfOutput(config)

    def forward(self, input_tensor, attention_mask, get_attention_matrices=False):
        self_output, attention_matrices = self.self(input_tensor, attention_mask, get_attention_matrices)
        attention_output = self.output(self_output, input_tensor)

        return attention_output, attention_matrices


class BertIntermediate(nn.Module):
    # 封装的FeedForward层和激活层
    def __init__(self, config):
        super(BertIntermediate, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        self.intermediate_act_fn = ACT2FN[config.hidden_act] # 使用哪种激活函数

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states) # 线性变换
        hidden_states = self.intermediate_act_fn(hidden_states) # 激活
        return hidden_states

class BertOutput(nn.Module):
    # 封装LayerNorm和残差连接，用于处理feedfoward层的输出
    def __init__(self, config):
        super(BertOutput, self).__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = BertLayerNorm(config.hidden_size, epsilon=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states

class BertLayer(nn.Module):
    # 一个Transformer block
    def __init__(self, config):
        super(BertLayer, self).__init__()
        self.attention = BertAttention(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(self, hidden_states, attention_mask, get_attention_matrices=False):
        # Attention层（包括LayerNorm和残差连接）
        attention_output, attention_matrices = self.attention(hidden_states, attention_mask, get_attention_matrices)
        # FeedForward层
        intermediate_output = self.intermediate(attention_output)
        # LayerNorm与残差输出层
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output, attention_matrices

class BertEncoder(nn.Module):
    # transformer block * N
    def __init__(self, config):
        super(BertEncoder, self).__init__()
        layer = BertLayer(config)
        # 复制N个transformer block
        self.layer = nn.ModuleList([copy.deepcopy(layer) for _ in range(config.num_hidden_layers)])

    def forward(self, hidden_states, attention_mask, output_all_encoded_layers=True, get_attention_matrices=False):
        '''
        :param output_all_encoded_layers: 是否输出每一个transformer block的隐藏层计算结果
        :param get_attention_matrices:    是否输出注意力矩阵，可用于可视化
        '''
        all_attention_matrices = []
        all_encoder_layers = []

        for layer_module in self.layer:
            hidden_states, attention_matrices = layer_module(hidden_states, attention_mask, get_attention_matrices)
            if output_all_encoded_layers:
                all_encoder_layers.append(hidden_states)
                all_attention_matrices.append(attention_matrices)
        if not output_all_encoded_layers:
            all_encoder_layers.append(hidden_states)
            all_attention_matrices.append(attention_matrices)

        return all_encoder_layers, all_attention_matrices

class BertPooler(nn.Module):
    '''
    Pooler是把隐藏层（hidden_state）中对应#CLS#的token的一条提取出来
    '''
    def __init__(self, config):
        super(BertPooler, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class BertPredictionHeadTransform(nn.Module):
    '''
    线性映射，激活，LayerNorm
    '''
    def __init__(self, config):
        super(BertPredictionHeadTransform, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.transform_act_fn = ACT2FN[config.hidden_act]
        self.LayerNorm = BertLayerNorm(config.hidden_size, epsilon=1e-12)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)

        return hidden_states

class BertLMPredictionHead(nn.Module):
    '''
    Masked Language Prediction Layer
    '''
    def __init__(self, config, bert_model_embedding_weights):
        super(BertLMPredictionHead, self).__init__()
        # 线性映射，激活，LayerNorm
        self.transfrom = BertPredictionHeadTransform(config)

        # 创建一个线性映射层，把transformer block输出的[batch_size, seq_len, embed_dim]
        # 映射为[batch_size, seq_len, vocab_size]
        # 这里其实可以直接矩阵层embedding矩阵的转置，但一般情况下我们要随机初始化新的一层参数
        # TODO
        self.decoder = nn.Linear(bert_model_embedding_weights.size(1),
                                 bert_model_embedding_weights.size(0),
                                 bias=False)
        self.decoder.weight = bert_model_embedding_weights
        self.bias = nn.Parameter(torch.zeros(bert_model_embedding_weights.size(0)))

    def forward(self, hidden_states):
        hidden_states = self.transfrom(hidden_states)
        hidden_states = self.decoder(hidden_states) + self.bias
        return hidden_states

class BertPreTrainingHeads(nn.Module):
    '''
    Bert的训练中通过隐藏层输出Masked LM的预测和Next Sentence的预测
    '''
    def __init__(self, config, bert_model_embedding_weights):
        super(BertPreTrainingHeads, self).__init__()

        # 把transformer block输出的[batch_size, seq_len, emb_dim]
        # 映射为[batch_size, seq_len, vocab_size], 用来进行MaskedLM预测
        self.predictions = BertLMPredictionHead(config, bert_model_embedding_weights)

        # 用来把pooled_output也就是#CLS#的那一条向量映射为2分类
        # 用来进行Next Sentence的预测
        self.seq_relationship = nn.Linear(config.hidden_size, 2)

    def forward(self, sequence_output, pooled_output):
        prediction_scores = self.predictions(sequence_output)
        seq_relationship_score = self.seq_relationship(pooled_output)
        return prediction_scores, seq_relationship_score

class BertPreTrainedModel(nn.Module):
    '''
    用来初始化模型参数
    '''
    def __init__(self, config, *inputs, **kwargs):
        super(BertPreTrainedModel, self).__init__()
        if not isinstance(config, BertConfig):
            raise ValueError('Parameter config is error!')

        self.config = config

    def init_bert_weights(self, module):
        '''
        Initialize the weights.
        '''
        if isinstance(module, (nn.Linear)):
            # 初始线性映射层的参数为正态分布
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, BertLayerNorm):
            # 初始化LayerNorm中alpha=1，beta=0
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

class BertModel(BertPreTrainedModel):
    def __init__(self, config):
        super(BertModel, self).__init__(config)
        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)
        self.pooler = BertPooler(config)
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, positional_enc, token_type_ids=None, attention_mask=None,
                output_all_encoded_layers=True, get_attention_matrices=False):
        '''
        :param input_ids: [batch_size, seq_len]
        :param positional_enc: [batch_size, seq_len, embedding_dim]
        :param token_type_ids: [batch_size, seq_len]. Type 0 corresponds to a 'sentence A' and type 1 corresponds to a 'sentence B';
        :param attention_mask: [batch_size, seq_len] with indices selected in [0, 1]
        '''

        # print("input_ids: \n", input_ids.numpy(), sep='', end='\n')
        # print("token_type_ids: \n", input_ids, sep='', end='\n')

        if attention_mask is None:
            attention_mask = (input_ids > 0) # [batch_size, seq_len]
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        # 注意力矩阵mask: [batch_size, 1, 1, seq_len]
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)

        # 给注意力矩阵里padding的无效区域加一个很大的负数的偏置，为了使softmax之后这些无效区域仍然为0，不参与后续计算
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        # embedding层
        embedding_output = self.embeddings(input_ids, positional_enc, token_type_ids)

        # 经过所有定义的transformer block之后的输出
        encoded_layers, all_attention_matrices = self.encoder(
            embedding_output,
            extended_attention_mask,
            output_all_encoded_layers,
            get_attention_matrices
        )

        # 可输出所有层的注意力矩阵用于可视化
        if get_attention_matrices:
            return all_attention_matrices
        # [-1]为最后一个transformer block的隐藏层的计算结果
        sequence_output = encoded_layers[-1]
        # pooled_output为隐藏层中#CLS#对应的token的一条向量
        pooled_output = self.pooler(sequence_output)
        if not output_all_encoded_layers:
            encoded_layers = encoded_layers[-1]
        return encoded_layers, pooled_output


class BertForPreTraining(BertPreTrainedModel):
    '''
    Bert Model with pre-training heads.
    This module comprises the Bert model followed by the two pre-training heads:
        - the masked language modeling head
        - the next sentence classification head.
    '''
    def __init__(self, config):
        super(BertForPreTraining, self).__init__(config)
        self.bert = BertModel(config)
        self.cls = BertPreTrainingHeads(config, self.bert.embeddings.word_embeddings.weight)
        self.apply(self.init_bert_weights)
        self.vocab_size = config.vocab_size
        self.next_loss_func = CrossEntropyLoss()
        self.mlm_loss_func = CrossEntropyLoss(ignore_index=0)

    def compute_loss(self, predictions, labels, num_class=2, ignor_index=-100):
        loss_func = CrossEntropyLoss(ignore_index=ignor_index)
        return loss_func(predictions.view(-1, num_class), labels.view(-1))

    def forward(self, input_ids, positional_enc, token_type_ids=None, attention_mask=None,
                masked_lm_labels=None, next_sentence_label=None):
        sequence_output, pooled_output = self.bert(input_ids, positional_enc, token_type_ids,
                                                   attention_mask, output_all_encoded_layers=False)
        mlm_preds, next_sentence_preds = self.cls(sequence_output, pooled_output)
        return mlm_preds, next_sentence_preds



