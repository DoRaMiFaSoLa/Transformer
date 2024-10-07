"""Summary:
    -- QKV matrixes learned by Two Convolutional Layer.
    -- Tip: Using AdatativeMaxPool to classifier.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from embedding import TransformerEmbeddings


device = torch.device('cuda')


class QKVAttention(nn.Module):
    """多头注意力QKV计算(运算进阶)。"""
    def __init__(self, head_size):
        super(QKVAttention, self).__init__()
        head_size = torch.tensor([head_size], dtype=torch.float32, device=device)
        self.sqrt_size = torch.sqrt(head_size)

    def forward(self, Q, K, V, valid_lens):
        """
        输入:
            Q:[batch_size, heads_num, seq_len, head_size]
            K:[batch_size, heads_num, seq_len, head_size]
            V:[batch_size, heads_num, seq_len, head_size]
            valid_len:[batch_size, seq_len]
        输出:
            context:[batch_size, heads_num, seq_len, head_size]
            attention_weights:[batch_size, heads_num, heads_num, seq_len, seq_len]
        """
        batch_size, heads_num, seq_len, head_size = Q.size()
        # Q:[batch_size, heads_num, 1, seq_len, head_size]
        Q = Q.reshape(batch_size, heads_num, 1, seq_len, head_size)
        # K:[batch_size, heads_num*heads_num, seq_len, head_size]
        K = K.repeat([1, heads_num, 1, 1])
        # K:[batch_size, heads_num, heads_num, seq_len, head_size]
        K = K.reshape(batch_size, heads_num, heads_num, seq_len, head_size)
        # score:[batch_size, heads_num, heads_num, seq_len, seq_len]
        score = torch.matmul(Q, K.transpose(3, 4)) / self.sqrt_size
        # attention_weights:[batch_size, heads_num, heads_num, seq_len, seq_len]
        attention_weights = F.softmax(score, -1)
        self._attention_weights = attention_weights
        # 加权平均
        # V:[batch_size, heads_num*heads_num, seq_len, head_size]
        V = V.repeat([1, heads_num, 1, 1])
        # K:[batch_size, heads_num, heads_num, seq_len, head_size]
        V = V.reshape(batch_size, heads_num, heads_num, seq_len, head_size)
        # B:[batch_size, heads_num, heads_num, seq_len, head_size]
        B = torch.matmul(attention_weights, V)
        # context:[batch_size, heads_num, seq_len, head_size]
        context = torch.sum(B, dim=2)
        # context:[batch_size, seq_len, heads_num, head_size]
        context = context.permute(0, 2, 1, 3)
        # context:[batch_size, seq_len, heads_num*head_size]
        context = context.reshape(batch_size, seq_len, heads_num*head_size)
        return context, attention_weights
    

class MultiHeadSelfAttention(nn.Module):
    """多头注意力。"""
    def __init__(self, inputs_size, heads_num, dropout=0.1):
        super(MultiHeadSelfAttention, self).__init__()
        self.heads_num = heads_num  # head的数目
        self.head_size = inputs_size // heads_num
        self.middle_size = int(4 * inputs_size / 3)
        assert(self.head_size * heads_num == inputs_size)
        self.Q_proj = nn.Sequential(nn.Conv2d(1, 8, 3, 1, 1).to(device),
                                    nn.BatchNorm2d(8).to(device),
                                    nn.ReLU().to(device),
                                    nn.Conv2d(8, 1, 3, 1, 1).to(device),
                                    nn.BatchNorm2d(1).to(device),
                                    nn.ReLU().to(device),
                                    )
        self.K_proj = nn.Sequential(nn.Conv2d(1, 8, 3, 1, 1).to(device),
                                    nn.BatchNorm2d(8).to(device),
                                    nn.ReLU().to(device),
                                    nn.Conv2d(8, 1, 3, 1, 1).to(device),
                                    nn.BatchNorm2d(1).to(device),
                                    nn.ReLU().to(device),
                                    )
        self.V_proj = nn.Sequential(nn.Conv2d(1, 8, 3, 1, 1).to(device),
                                    nn.BatchNorm2d(8).to(device),
                                    nn.ReLU().to(device),
                                    nn.Conv2d(8, 1, 3, 1, 1).to(device),
                                    nn.BatchNorm2d(1).to(device),
                                    nn.ReLU().to(device),
                                    )
        self.out_proj = nn.Sequential(nn.Conv2d(1, 8, 3, 1, 1).to(device),
                                    nn.BatchNorm2d(8).to(device),
                                    nn.ReLU().to(device),
                                    nn.Conv2d(8, 1, 3, 1, 1).to(device),
                                    nn.BatchNorm2d(1).to(device),
                                    nn.ReLU().to(device),
                                    )
        self.attention = QKVAttention(self.head_size)

    def split_head_reshape(self, X, heads_num, head_size):
        """
        输入:
            X:[batch_size, 1, seq_len, hidden_size]
        输出:
            output:[batch_size, heads_num, seq_len, head_size]
        """
        batch_size, i, seq_len, hidden_size = X.shape
        # X:[batch_size, seq_len, hidden_size]
        X = X.reshape(batch_size*i, seq_len, hidden_size)
        # 多头重组
        # X:[batch_size, seq_len, heads_num, head_size]
        X = torch.reshape(X, shape=[batch_size, seq_len, heads_num, head_size])
        # 形状重组
        # X:[batch_size, heads_num, seq_len, head_size]
        X = X.permute(0, 2, 1, 3)
        return X 


    def forward(self, X, valid_lens):
        """
        输入:
            X:输入矩阵, shape=[batch_size,seq_len,hidden_size]
            valid_lens:长度矩阵,shape=[batch_size]
        输出:
            output:输出矩阵, 表示的是多头注意力的结果
        """
        self.batch_size, self.seq_len, self.hidden_size = X.shape
        # X:[batch_size, 1, seq_len, hidden_size]
        X = X.reshape(self.batch_size, 1, self.seq_len, self.hidden_size)
        # Q, K, V:[batch_size, 1, seq_len, hidden_size]
        Q, K, V = self.Q_proj(X), self.K_proj(X), self.V_proj(X)
        # Q, K, V:[batch_size, heads_num, seq_len, head_size]
        Q = self.split_head_reshape(Q, self.heads_num, self.head_size)
        K = self.split_head_reshape(K, self.heads_num, self.head_size)
        V = self.split_head_reshape(V, self.heads_num, self.head_size)
        # out:[batch_size, seq_len, heads_num*head_size]
        out, atten_weights = self.attention(Q, K, V, valid_lens)
        batch_size, seq_len, hidden_size = out.shape
        # out:[batch_size, 1, seq_len, heads_num*head_size]
        out = out.reshape(batch_size, 1, seq_len, hidden_size)
        out = self.out_proj(out)
        # out:[batch_size, seq_len, heads_num*head_size]
        out = out.reshape(batch_size, seq_len, hidden_size)
        return out, atten_weights


class PositionwiseFeedForward(nn.Module):
    """逐位前馈层。"""
    def __init__(self, input_size, mid_size, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.features = nn.Sequential(
            nn.Linear(input_size, mid_size).to(device),
            nn.ReLU().to(device),
            nn.Dropout(dropout).to(device),
            nn.Linear(mid_size, input_size).to(device)
        )

    def forward(self, X):
        out = self.features(X)
        return out


class AddNorm(nn.Module):
    """加与规范化。"""
    def __init__(self, size, dropout):
        super(AddNorm, self).__init__()
        self.layer_norm = nn.LayerNorm(size).to(device)
        self.dropout = nn.Dropout(dropout).to(device)

    def forward(self, X, H):
        H = X + self.dropout(H)
        out = self.layer_norm(H)
        return out
    

class TransformerBlock(nn.Module):
    """Transformer 模块。"""
    def __init__(self, input_size, head_num, pff_size, an_dropout=0.1, attention_dropout=None, pff_dropout=None):
        """
        input_size:输入数据维度。
        head_num:多头自注意力多头个数。
        pff_size:逐位前馈层的大小。
        an_dropout:加与规范化 dropout 参数。
        attn_dropout:多头注意力的 dropout 参数。
        ppf_dropout:逐位前馈层的 dropout 参数。 
        """
        super(TransformerBlock, self).__init__()
        self.attention_dropout = an_dropout if attention_dropout is None else attention_dropout  # 多头注意力里面的 Dropout参数
        self.pff_dropout = an_dropout if pff_dropout is None else pff_dropout  # 逐位前馈层里面的 Dropout参数
        # 多头自注意力机制
        self.multi_head_attention = MultiHeadSelfAttention(input_size, head_num, dropout=self.attention_dropout)
        self.pff = PositionwiseFeedForward(input_size, pff_size, dropout=self.pff_dropout)  # 逐位前馈层
        self.addnorm = AddNorm(input_size, an_dropout)   # 加与规范化

    def forward(self, X, src_mask=None):
        X_atten, attention_weights = self.multi_head_attention(X, src_mask)  # 多头注意力
        X = self.addnorm(X, X_atten)  # 加与规范化
        X_pff = self.pff(X)  # 前馈层
        X = self.addnorm(X, X_pff)  # 加与规范化
        return X, attention_weights
    

class Transformer(nn.Module):
    """Transformer Encoder 复现。"""
    def __init__(self, vocab_size, n_block=2, hidden_size=768, heads_num=4, intermediate_size=3072, hidden_dropout=0.1, attention_dropout=0.1, 
                pff_dropout=0, position_size=512, num_classes=2, padding_idx=0):
        super(Transformer, self).__init__()
        """
        vocab_size:词表大小。
        n_block:Transformer 编码器数目。
        hidden_size:每个词映射成稠密向量的个数。
        heads_num:多头注意力的个数。
        intermediate_size:逐位前馈层的维度。
        hidden_dropout:Embedding 层的dropout。
        attention_dropout:多头注意力的 dropout。
        position_size:位置编码大小。
        num_classes:类别数。
        pff_dropout:逐位前馈层的 dropout。
        padding_idx:填充字符的id。
        """
        self.padding_idx = padding_idx
        # 嵌入层
        self.embeddings = TransformerEmbeddings(vocab_size, hidden_size, hidden_dropout, position_size)
        # Transformer的编码器
        self.layers = []
        for _ in range(n_block):
            encoder_layer = TransformerBlock(hidden_size, heads_num, intermediate_size, an_dropout=hidden_dropout, 
                                            attention_dropout=attention_dropout, pff_dropout=pff_dropout)
            self.layers.append(encoder_layer)
        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),  # 全连接层
            nn.Tanh(),  # 双曲正切激活函数
            nn.Linear(hidden_size, num_classes)  # 最后一层分类器
        )

    def forward(self, input_ids, segment_ids, position_ids=None, attention_mask=None):
        # 构建Mask矩阵
        if attention_mask is None:
            attention_mask = (input_ids == self.padding_idx) * -1e9
            attention_mask = attention_mask.float()
        # 抽取特征向量
        embedding_output = self.embeddings(input_ids=input_ids, position_ids=position_ids, segment_ids=segment_ids)
        sequence_output = embedding_output
        self._attention_weights = []
        # Transformer的输出和注意力权重的输出
        for i, encoder_layer in enumerate(self.layers):
            sequence_output, atten_weights = encoder_layer(sequence_output, src_mask=attention_mask)
            self._attention_weights.append(atten_weights)
        # 选择第0个位置的向量作为句向量
        first_token_tensor = sequence_output[:, 0]
        # 分类器
        logits = self.classifier(first_token_tensor)
        return logits
    
    @property
    def attention_weights(self):
        return self._attention_weights
