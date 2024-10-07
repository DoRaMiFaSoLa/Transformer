import torch
import torch.nn as nn
import numpy as np


class WordEmbedding(nn.Module):
    """输入编码。"""
    def __init__(self, vocab_size, emb_size, padding_idx=0):
        super(WordEmbedding, self).__init__()
        # Embedding的维度
        self.emb_size = emb_size
        # 使用随机高斯分布初始化 embedding
        self.word_embedding = nn.Embedding(vocab_size, emb_size, padding_idx=padding_idx)
        self.word_embedding.weight.data.normal_(0.0, self.emb_size ** -0.5) 

    def forward(self, word):
        word_emb = self.word_embedding(word)
        word_emb = self.emb_size ** 0.5 * word_emb
        return word_emb
    

class SegmentEmbedding(nn.Module):
    """分段编码。"""
    def __init__(self, segment_size, emb_size):
        super(SegmentEmbedding, self).__init__()
        # Embedding的维度
        self.emb_size = emb_size
        # 分段编码
        self.seg_embedding = nn.Embedding(segment_size, emb_size)
        self.seg_embedding.weight.data.normal_(0.0, self.emb_size ** -0.5)

    def forward(self, word):
        seg_embedding = self.seg_embedding(word)
        seg_embedding = self.emb_size ** 0.5 * seg_embedding
        return seg_embedding
    

def get_sinusoid_encoding(position_size, hidden_size):
    def cal_angle(pos, hidden_idx):
        # i = hid_idx // 2
        return pos / np.power(10000, 2 * (hidden_idx // 2) / hidden_size)

    def get_posi_angle_vec(pos):
        return [cal_angle(pos, hidden_j) for hidden_j in range(hidden_size)]

    sinusoid = np.array([get_posi_angle_vec(pos_i) for pos_i in range(position_size)])
    sinusoid[:, 0::2] = np.sin(sinusoid[:, 0::2])
    sinusoid[:, 1::2] = np.cos(sinusoid[:, 1::2])
    # position_size × hidden_size  得到每一个词的位置向量
    return sinusoid.astype("float32")


class PositionalEmbedding(nn.Module):
    """位置编码。"""
    def __init__(self, max_length, emb_size):
        super(PositionalEmbedding, self).__init__()
        self.emb_size = emb_size
        # 使用三角函数初始化Embedding
        self.pos_encoder = nn.Embedding(max_length, emb_size)
        encoding = get_sinusoid_encoding(max_length, emb_size)
        encoding = torch.from_numpy(encoding)
        self.pos_encoder.weight.data.copy_(encoding)
    
    def forward(self, pos):
        pos_emb = self.pos_encoder(pos)
        # 关闭位置编码的梯度更新
        pos_emb = pos_emb.detach()
        pos_emb.requires_grad_(False)
        return pos_emb
    

class TransformerEmbeddings(nn.Module):
    """包括输入编码，分段编码，位置编码。"""
    def __init__(self, vocab_size, hidden_size=768, hidden_dropout_prob=0.1, position_size=512, segment_size=2):
        """
        vocab_size:词表大小。
        hidden_size:高维维度。
        hidden_dropout_prob:dropout 概率。
        position_size:位置编码最大长度。
        segment_size:分割词表大小。
        """
        super(TransformerEmbeddings, self).__init__()
        self.word_embeddings = WordEmbedding(vocab_size, hidden_size)  # 输入编码向量
        self.position_embeddings = PositionalEmbedding(position_size, hidden_size)  # 位置编码向量
        self.segment_embeddings = SegmentEmbedding(segment_size, hidden_size)  # 分段编码
        self.layer_norm = nn.LayerNorm(hidden_size)  # 层规范化
        self.dropout = nn.Dropout(hidden_dropout_prob)

    def forward(self, input_ids, segment_ids = None, position_ids = None):
        if position_ids is None:
            # 初始化全1的向量，比如[1,1,1,1]
            ones = torch.ones_like(input_ids, dtype=torch.int64)
            # 累加输入,求出序列前K个的长度,比如[1,2,3,4]
            seq_length = torch.cumsum(ones, dim=-1)
            # position id的形式： 比如[0,1,2,3]
            position_ids = seq_length - ones
            position_ids.requires_grad_(False)
        input_embeddings = self.word_embeddings(input_ids)
        segment_embeddings = self.segment_embeddings(segment_ids)
        position_embeddings = self.position_embeddings(position_ids)
        embeddings = input_embeddings + segment_embeddings + position_embeddings
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings