import torch
import utils
from torch import nn
from data import truncate_pad
import math
import collections

class EncoderDecoder(nn.Module):
    def __init__(self, encoder, decoder, **kwargs):
        """
        The base class for the encoder-decoder architecture.
        
        Parameters
        ----------
        @param encoder : nn.Module
            The encoder
        @param decoder : nn.Module
            The decoder
        """
        super(EncoderDecoder, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, enc_X, dec_X, *args):
        """
        Parameters
        ----------
        @param enc_X : torch.Tensor
            Shape (batch_size, source_seq_len)
        @param dec_X : torch.Tensor
            Shape (batch_size, target_seq_len)
        @param args : list
            Additional arguments, such as the valid length
        """
        enc_outputs = self.encoder(enc_X, *args)
        dec_state = self.decoder.init_state(enc_outputs, *args)
        return self.decoder(dec_X, dec_state)


def sequence_mask(X, valid_len, value=0):
    """
    Mask irrelevant entries in sequences.

    Parameters
    ----------
    @param X : torch.Tensor
        Shape (batch_size, seq_len, vocab_size)
    @param valid_len : torch.Tensor
        Shape (batch_size, )
    @param value : float
        The value to be substituted in place of the masked values
    @return X : torch.Tensor
        Shape (batch_size, seq_len, vocab_size)
    """
    maxlen = X.size(1)
    mask = torch.arange((maxlen), dtype=torch.float32,
                        device=X.device)[None, :] < valid_len[:, None]
    X[~mask] = value
    return X

class MaskedSoftmaxCELoss(nn.CrossEntropyLoss):
    def forward(self, pred, label, valid_len):
        """
        The softmax cross-entropy loss with masks.

        Parameters
        ----------
        @param pred : torch.Tensor
            Shape (batch_size, seq_len, num_classes)
        @param label : torch.Tensor
            Shape (batch_size, seq_len)
        @param valid_len : torch.Tensor
            Shape (batch_size, )
        @return weighted_loss : torch.Tensor
            The loss weighted by the mask
        """
        weights = torch.ones_like(label)
        weights = sequence_mask(weights, valid_len)
        self.reduction='none'
        unweighted_loss = super(MaskedSoftmaxCELoss, self).forward(
            pred.permute(0, 2, 1), label)
        weighted_loss = (unweighted_loss * weights).mean(dim=1)
        return weighted_loss

def grad_clipping(net, theta):
    """
    Clip the gradient.
    
    Parameters
    ----------
    @param net : nn.Module
        The network to train
    @param theta : float
        The clipping threshold
    """
    if isinstance(net, nn.Module):
        params = [p for p in net.parameters() if p.requires_grad]
    else:
        params = net.params
    norm = torch.sqrt(sum(torch.sum((p.grad ** 2)) for p in params))
    if norm > theta:
        for param in params:
            param.grad[:] *= theta / norm

def train_seq2seq(net, data_iter, lr, num_epochs, tgt_vocab, device):
    """
    Train a model for sequence to sequence.

    Parameters
    ----------
    @param net : nn.Module
        The network to train
    @param data_iter : DataIterator
        The data iterator
    @param lr : float
        Learning rate
    @param num_epochs : int
        Number of epochs
    @param tgt_vocab : Vocab
        The vocabulary of the target language
    @param device : torch.device
        The device to run the training on
    """
    def xavier_init_weights(m):
        if type(m) == nn.Linear:
            nn.init.xavier_uniform_(m.weight)
        if type(m) == nn.GRU:
            for param in m._flat_weights_names:
                if "weight" in param:
                    nn.init.xavier_uniform_(m._parameters[param])
    net.apply(xavier_init_weights)
    net.to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    loss = MaskedSoftmaxCELoss()
    net.train()
    animator = utils.Animator(xlabel='epoch', ylabel='loss',
                            xlim=[10, num_epochs])
    for epoch in range(num_epochs):
        timer = utils.Timer()
        metric = utils.Accumulator(2)  # Sum of training loss, no. of tokens
        for batch in data_iter:
            optimizer.zero_grad()
            X, X_valid_len, Y, Y_valid_len = [x.to(device) for x in batch]
            bos = torch.tensor([tgt_vocab['<bos>']] * Y.shape[0],
                               device=device).reshape(-1, 1)
            dec_input = torch.cat([bos, Y[:, :-1]], 1)  # Teacher forcing
            Y_hat, _ = net(X, dec_input, X_valid_len)
            l = loss(Y_hat, Y, Y_valid_len)
            l.sum().backward()  # Make the loss scalar for `backward`
            grad_clipping(net, 1)
            num_tokens = Y_valid_len.sum()
            optimizer.step()
            with torch.no_grad():
                metric.add(l.sum(), num_tokens)
        if (epoch + 1) % 10 == 0:
            animator.add(
                epoch + 1, (metric[0] / metric[1],)
            )
    print(f'loss {metric[0] / metric[1]:.3f}, {metric[1] / timer.stop():.1f} '
          f'tokens/sec on {str(device)}')
    
def predict_seq2seq(net, src_sentence, src_vocab, tgt_vocab, num_steps,
                    device, save_attention_weights=False):
    """
    Predict for sequence to sequence.

    Parameters
    ----------
    @param net : nn.Module
        The trained model
    @param src_sentence : str
        The source sentence
    @param src_vocab : Vocab
        The source vocabulary
    @param tgt_vocab : Vocab
        The target vocabulary
    @param num_steps : int
        The length of the generated sequence minus 1
    @param device : torch.device
        The device to run the prediction on
    @param save_attention_weights : bool, default False
        Whether to save attention weights to the attention_weights list
    @return : str
        The generated sequence
    """
    # Set `net` to eval mode for inference
    net.eval()
    src_tokens = src_vocab[src_sentence.lower().split(' ')] + [
        src_vocab['<eos>']]
    enc_valid_len = torch.tensor([len(src_tokens)], device=device)
    src_tokens = truncate_pad(src_tokens, num_steps, src_vocab['<pad>'])
    # Add the batch axis
    enc_X = torch.unsqueeze(
        torch.tensor(src_tokens, dtype=torch.long, device=device), dim=0)
    enc_outputs = net.encoder(enc_X, enc_valid_len)
    dec_state = net.decoder.init_state(enc_outputs, enc_valid_len)
    # Add the batch axis
    dec_X = torch.unsqueeze(torch.tensor(
        [tgt_vocab['<bos>']], dtype=torch.long, device=device), dim=0)
    output_seq, attention_weight_seq = [], []
    for _ in range(num_steps):
        Y, dec_state = net.decoder(dec_X, dec_state)
        # We use the token with the highest prediction likelihood as the input
        # of the decoder at the next time step
        dec_X = Y.argmax(dim=2)
        pred = dec_X.squeeze(dim=0).type(torch.int32).item()
        # Save attention weights (to be covered later)
        if save_attention_weights:
            attention_weight_seq.append(net.decoder.attention_weights)
        # Once the end-of-sequence token is predicted, the generation of the
        # output sequence is complete
        if pred == tgt_vocab['<eos>']:
            break
        output_seq.append(pred)
    return ' '.join(tgt_vocab.to_tokens(output_seq)), attention_weight_seq


def bleu(pred_seq, label_seq, k):
    """
    Compute the BLEU.
    
    Parameters
    ----------
    @param pred_seq : str
        The predicted sequence
    @param label_seq : str
        The label sequence
    @param k : int
        The maximum number of tokens in a predicted sequence
    """
    # 将预测序列和标签序列分别按空格分割成词元列表
    pred_tokens, label_tokens = pred_seq.split(' '), label_seq.split(' ')
    # 计算预测序列和标签序列的长度
    len_pred, len_label = len(pred_tokens), len(label_tokens)
    # 计算惩罚因子，当预测序列长度小于标签序列长度时，惩罚因子为1，否则为e^(1-len_label/len_pred)
    score = math.exp(min(0, 1 - len_label / len_pred))
    # 计算n-gram的匹配数和出现次数
    for n in range(1, k + 1):
        num_matches, label_subs = 0, collections.defaultdict(int)
        # 统计标签序列中n-gram的出现次数
        for i in range(len_label - n + 1):
            label_subs[' '.join(label_tokens[i: i + n])] += 1
        # 统计预测序列中n-gram的匹配数
        for i in range(len_pred - n + 1):
            if label_subs[' '.join(pred_tokens[i: i + n])] > 0:
                num_matches += 1
                label_subs[' '.join(pred_tokens[i: i + n])] -= 1
        # 计算n-gram的精确度和权重
        score *= math.pow(num_matches / (len_pred - n + 1), math.pow(0.5, n))
    # 返回BLEU得分
    return score