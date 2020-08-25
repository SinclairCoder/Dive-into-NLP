import collections
import os
import io
import math
import torch
from torch import nn
import torch.nn.functional as F
import torchtext.vocab as Vocab
import torch.utils.data as Data
import sys

PAD, BOS, EOS = '<pad>','<bos>','<eos>'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 将一个序列中所有的词记录在all_tokens中以便之后构造词典，然后在该序列后面添加PAD直到序列
# 长度变为max_seq_len，然后将序列保存在all_seqs中
def process_one_seq(seq_tokens,all_tokens,all_seqs,max_seq_len):
    all_tokens.extend(seq_tokens)
    seq_tokens += [EOS]+[PAD]*(max_seq_len-len(seq_tokens)-1)
    all_seqs.append(seq_tokens)

# 使用所有的词来构造词典，并将所有序列中的词变换为词索引后构造Tensor
def build_data(all_tokens,all_seqs):
    vocab = Vocab.Vocab(collections.Counter(all_tokens),specials=[PAD,BOS,EOS])
    indices = [[vocab.stoi[w] for w in seq] for seq in all_seqs]
    return vocab, torch.tensor(indices)

def read_data(max_seq_len):
    in_tokens,out_tokens,in_seqs,out_seqs = [],[],[],[]
    with io.open('fr-en-small.txt') as f:
        lines = f.readlines()
    for line in lines:
        in_seq,out_seq = line.rstrip().split('\t')
        in_seq_tokens,out_seq_tokens = in_seq.split(' '), out_seq.split(' ')
        if max(len(in_seq_tokens),len(out_seq_tokens)) > max_seq_len-1:
            continue
        process_one_seq(in_seq_tokens,in_tokens,in_seqs,max_seq_len)
        process_one_seq(out_seq_tokens,out_tokens,out_seqs,max_seq_len)
    in_vocab, in_data = build_data(in_tokens,in_seqs)
    out_vocab,out_data = build_data(out_tokens,out_seqs)
    return in_vocab,out_vocab,Data.TensorDataset(in_data,out_data)


class Encoder(nn.Module):
    def __init__(self,vocab_size,embed_size,num_hiddens,num_layers,drop_prob=0,**kwargs):
        super(Encoder,self).__init__(**kwargs)
        self.embedding = nn.Embedding(vocab_size,embed_size)
        # Inputs: input, h_0  Outputs: output, h_n
        self.rnn = nn.GRU(embed_size,num_hiddens,num_layers,dropout=drop_prob)
    def forward(self,inputs,state):
        # 输入维度：batch_size* time
        # 将[seq_len, batch_size, feture_size] 变为 [batch_size, seq_len, feature_size]
        embedding = self.embedding(inputs.long()).permute(1,0,2)
        # Inputs: input, h_0  Outputs: output, h_n
        return self.rnn(embedding,state)
    def begin_state(self):
        return None
    
# 最初的Attention文章中函数a就是将输入连结后通过含单隐藏层的多层感知机变换
def attention_model(input_size,attention_size):
    model = nn.Sequential(nn.Linear(input_size,attention_size,bias=False),
                          nn.Tanh(),
                          nn.Linear(attention_size,1,bias=False))
    return model
def attention_forward(model,enc_states,dec_state):
    """
        enc_states: (时间步数，批量大小，隐藏单元个数)
        dec_state: (批量大小，隐藏单元个数)
    """
    # 将解码器隐藏状态广播到和编码器隐藏状态形状相同后进行连结
    dec_states = dec_state.unsqueeze(dim=0).expand_as(enc_states)
    enc_and_dec_states = torch.cat((enc_states,dec_states),dim=2)
    e = model(enc_and_dec_states)
    alpha = F.softmax(e,dim=0)
    return (alpha*enc_states).sum(dim=0)

class Decoder(nn.Module):
    def __init__(self,vocab_size,embed_size,num_hiddens,num_layers,attention_size,drop_prob=0):
        super(Decoder,self).__init__()
        self.embedding = nn.Embedding(vocab_size,embed_size)
        self.attention = attention_model(2*num_hiddens,attention_size)
         # GRU的输入包含attention输出的c和实际输入, 所以尺寸是 num_hiddens+embed_size
        self.rnn = nn.GRU(num_hiddens+embed_size,num_hiddens,num_layers,dropout=drop_prob)
        self.out = nn.Linear(num_hiddens,vocab_size)
    def forward(self,cur_input,state,enc_states):
        """
        cur_input shape: (batch, )
        state shape: (num_layers, batch, num_hiddens)
        """
         # 使用注意力机制计算背景向量
        c = attention_forward(self.attention,enc_states,state[-1])
        # 将嵌入后的输入和背景向量在特征维连结, (批量大小, num_hiddens+embed_size)
        input_and_c = torch.cat((self.embedding(cur_input),c),dim=1)
        # 为输入和背景向量的连结增加时间步维，时间步个数为1
        output, state = self.rnn(input_and_c.unsqueeze(0),state)
        # 移除时间步维，输出形状为(批量大小, 输出词典大小)
        output = self.out(output).squeeze(dim=0)
        return output,state
    def begin_state(self,enc_state):
        return enc_state
    
def batch_loss(encoder,decoder,X,Y,loss):
    batch_size = X.shape[0]
    enc_state = encoder.begin_state()
    enc_outputs,enc_state = encoder(X,enc_state)
    dec_state = decoder.begin_state(enc_state)
    dec_input = torch.tensor([out_vocab.stoi[BOS]]*batch_size)
    # 将使用掩码变量mask来忽略掉标签为填充项PAD的损失
    mask,num_not_pad_tokens = torch.ones(batch_size,),0
    l = torch.tensor([0.0])
    for y in Y.permute(1,0):  # Y shape: (batch, seq_len)
        dec_output, dec_state = decoder(dec_input,dec_state,enc_outputs)
        l = l+(mask*loss(dec_output,y)).sum()
        dec_input = y
        num_not_pad_tokens += mask.sum().item()
        mask = mask*(y!=out_vocab.stoi[EOS]).float()
    return l/num_not_pad_tokens

def train(encoder,decoder,dataset,lr,batch_size,num_epochs):
    enc_optimizer = torch.optim.Adam(encoder.parameters(),lr=lr)
    dec_optimizer = torch.optim.Adam(decoder.parameters(),lr = lr)
    
    loss = nn.CrossEntropyLoss(reduction='none')
    data_iter = Data.DataLoader(dataset,batch_size,shuffle=True)
    for epoch in range(num_epochs):
        l_sum = 0.0
        for X,Y in data_iter:
            enc_optimizer.zero_grad()
            dec_optimizer.zero_grad()
            l = batch_loss(encoder,decoder,X,Y,loss)
            l.backward()
            enc_optimizer.step()
            dec_optimizer.step()
            l_sum += l.item()
        if (epoch+1)%10==0:
              print("epoch %d, loss %.3f" % (epoch + 1, l_sum / len(data_iter)))

def translate(encoder,decoder,input_seq,max_seq_len):
    in_tokens = input_seq.split(' ')
    in_tokens += [EOS] + [PAD] *(max_seq_len-len(in_tokens)-1)
    # batch_size = 1
    enc_input = torch.tensor([[in_vocab.stoi[tk] for tk in in_tokens]])
    enc_state = encoder.begin_state()
    enc_output,enc_state = encoder(enc_input,enc_state)
    dec_input = torch.tensor([out_vocab.stoi[BOS]])
    dec_state = decoder.begin_state(enc_state)
    output_tokens = []
    for _ in range(max_seq_len):
        dec_output, dec_state = decoder(dec_input,dec_state,enc_output)
        pred = dec_output.argmax(dim=1)
        pred_token = out_vocab.itos[int(pred.item())]
        if pred_token == EOS:
            break
        else :
            output_tokens.append(pred_token)
            dec_input = pred
    return output_tokens


def bleu(pred_tokens,label_tokens,k):
    len_pred, len_label = len(pred_tokens), len(label_tokens)
    score = math.exp(min(0,1-len_label/len_pred))
    for n in range(1,k+1):
        num_matches, label_subs = 0, collections.defaultdict(int)
        for i in range(len_label-n+1):
            label_subs[''.join(label_tokens[i:i+n])]+=1
        for i in range(len_pred-n+1):
            if label_subs[''.join(pred_tokens[i:i+n])]>0:
                num_matches += 1
                label_subs[''.join(pred_tokens[i:i+n])] -= 1
        score *= math.pow(num_matches/(len_pred-n+1),math.pow(0.5,n))
    return score
def score(input_seq,label_seq,k):
    pred_tokens = translate(encoder,decoder,input_seq,max_seq_len)
    label_tokens = label_seq.split(' ')
    print('bleu %.3f, predict: %s'%(bleu(pred_tokens,label_tokens,k),' '.join(pred_tokens)))
    
    
if __name__ == '__main__':
    
    embed_size,num_hiddens,num_layers = 64,64,2
    attention_size, drop_prob, lr, batch_size,num_epochs = 10,0.5,0.01,2,50
    encoder = Encoder(len(in_vocab),embed_size,num_hiddens,num_layers,drop_prob)
    decoder = Decoder(len(out_vocab),embed_size,num_hiddens,num_layers,attention_size,drop_prob)
    train(encoder,decoder,dataset,lr,batch_size,num_epochs)
    input_seq = 'ils regardent .'
    translate(encoder, decoder, input_seq, max_seq_len)
    score('ils regardent .', 'they are watching .', k=2)
    score('ils sont canadienne .', 'they are canadian .', k=2)
    
    