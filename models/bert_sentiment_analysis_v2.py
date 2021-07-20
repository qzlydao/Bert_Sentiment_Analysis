from models.bert_model import *

class Bert_Sentiment_Analysis(nn.Module):
    '''
    使用mean-max pool的方式进行情感分析
    '''
    def __init__(self, config):
        super(Bert_Sentiment_Analysis, self).__init__()
        self.bert = BertModel(config)
        # 因为要拼接mean和max两个向量，所以是2*hidden_size
        self.dense = nn.Linear(config.hidden_size * 2, config.hidden_size)
        self.final_dense = nn.Linear(config.hidden_size, 1)
        self.activate = nn.Sigmoid()

    def compute_loss(self, predictions, labels):
        # 将预测和标记的维度展平，防止出现维度不一致
        # todo predictions维度 [batch_size]
        predictions = predictions.view(-1)
        labels = labels.float().view(-1)
        epsilon = 1e-8
        # 交叉熵, 二分类的交叉熵
        loss = - labels * torch.log(predictions + epsilon) - \
               (torch.tensor(1.) - labels) * torch.log(torch.tensor(1.) - predictions + epsilon)

        loss = torch.mean(loss)
        return loss

    def forward(self, text_input, positional_enc, labels=None):
        # bert返回 (encoded_layers, pooled_output) pooled_output用于next_sentence_prediction
        encoded_layers, _ = self.bert(text_input, positional_enc,
                                      output_all_encoded_layers=True)
        # todo 为什么是第三层的encoded_layer
        # sequence_output [batch_size, seq_len, embed_dim]
        sequence_output = encoded_layers[2]

        # [batch_size, embed_dim]
        avg_pooled = sequence_output.mean(1)

        # [batch_size, embed_dim]
        max_pooled = torch.max(sequence_output, dim=1)

        # mean-max拼接, 拼接后 [batch_size, embed*2]
        pooled = torch.cat((avg_pooled, max_pooled[0]), dim=1)
        pooled = self.dense(pooled)

        # [batch_size, embed_dim*2] -> [batch_size, 1]
        predictions = self.final_dense(pooled)

        # 用sigmoid激活函数
        predictions = self.activate(predictions)

        if labels is not None:
            # 计算loss
            loss = self.compute_loss(predictions, labels)
            return predictions, loss
        else:
            return predictions





if __name__ == '__main__':

    a = torch.randint(0, 5, (3, 2, 4)).float()
    print(a)
    print()

    avg_pooled = a.mean(1)
    max_pooled = torch.max(a, dim=1)


    print(avg_pooled)
    print()
    print(max_pooled[0])
    print()

    pooled = torch.cat((avg_pooled, max_pooled[0]), dim=1)
    print(pooled)


    pass