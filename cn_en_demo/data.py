import os
import collections
import torch
from torch import nn

def read_en_train(data_dir = "./wnt_cn_en/data/train"):
    """
    载入英文数据集
    
    Parameters
    ----------
    @param data_dir: str
    @return: str
    """
    with open(os.path.join(data_dir, 'news-commentary-v13.zh-en.en'), 'r',
             encoding='utf-8') as f:
        return f.read()
    
def read_cn_train(data_dir = "./wnt_cn_en/data/train"):
    """
    载入中文数据集
    
    Parameters
    ----------
    @param data_dir: str
    @return: str
    """
    with open(os.path.join(data_dir, 'news-commentary-v13.zh-en.zh'), 'r',
             encoding='utf-8') as f:
        return f.read()
    

def tokenize(text, num_examples=None):
    """
    词元化

    Parameters
    ----------
    @param text: str
        The text to tokenize
    @param num_examples: int
        The number of examples in the dataset
    @return: list
        The tokenized text 2D list
    """
    tokens = []
    for i, line in enumerate(text.split('\n')):
        if num_examples and i > num_examples:
            break
        if line:
            tokens.append(line.split(' '))
    return tokens


class Vocab:
    def __init__(self, tokens=None, min_freq=0, reserved_tokens=None):
        """
        Vocabulary for text.
        
        Parameters
        ----------
        @param tokens: list of token lists
        @param min_freq: int
            The minimum frequency required for a token to be included in the vocabulary
        @param reserved_tokens: list of str
            The token list that will be added to the vocabulary
        """
        if tokens is None:
            tokens = []
        if reserved_tokens is None:
            reserved_tokens = []
        # Sort according to frequencies
        counter = count_corpus(tokens)
        self._token_freqs = sorted(
            counter.items(),
            key=lambda x: x[1],
            reverse=True
        )
        # The index for the unknown token is 0
        self.idx_to_token = ['<unk>'] + reserved_tokens
        self.token_to_idx = {token: idx
                             for idx, token in enumerate(self.idx_to_token)}
        for token, freq in self._token_freqs:
            if freq < min_freq:
                break
            if token not in self.token_to_idx:
                self.idx_to_token.append(token)
                self.token_to_idx[token] = len(self.idx_to_token) - 1

    def __len__(self):
        return len(self.idx_to_token)

    def __getitem__(self, tokens):
        if not isinstance(tokens, (list, tuple)):
            return self.token_to_idx.get(tokens, self.unk)
        return [self.__getitem__(token) for token in tokens]

    def to_tokens(self, indices):
        if not isinstance(indices, (list, tuple)):
            return self.idx_to_token[indices]
        return [self.idx_to_token[index] for index in indices]

    @property
    def unk(self):  # Index for the unknown token
        return 0

    @property
    def token_freqs(self):  # Index for the unknown token
        return self._token_freqs


def count_corpus(tokens):
    """
    Count token frequencies.
    
    Parameters
    ----------
    @param tokens: list of token lists
    @return: `collections.Counter` instance that maps tokens to frequencies
    """
    # Here `tokens` is a 1D list or 2D list
    if len(tokens) == 0 or isinstance(tokens[0], list):
        # Flatten a list of token lists into a list of tokens
        tokens = [token for line in tokens for token in line]
    return collections.Counter(tokens)

def truncate_pad(line, num_steps, padding_token):
    """
    截断或填充文本序列
    
    Parameters
    ----------
    @param line: list
        The text sequence after tokenization
    @param num_steps: int
        The length of the resulting sequence
    @param padding_token: str
        The padding token
    @return: list
        The truncated or padded text sequence
    """
    if len(line) > num_steps:
        return line[:num_steps]  # 截断
    return line + [padding_token] * (num_steps - len(line))  # 填充

def build_array(tokens, vocab, num_steps):
    """
    将文本序列转换成向量
    
    Parameters
    ----------
    @param tokens: list of token lists
        2D list 
    @param vocab: instance of Vocab
    @param num_steps: int
        The length of the sequence of tokens in a batch
    @return: tuple of (array, valid_len)
        - array: tensor of shape (batch_size, num_steps)
        - valid_len: tensor of shape (batch_size, )
    """
    tokens = [vocab[token_line] for token_line in tokens]
    tokens = [token_line + [vocab['<eos>']] for token_line in tokens]
    array = torch.tensor([
        truncate_pad(
            line=token_line, 
            num_steps=num_steps, 
            padding_token=vocab['<pad>']
        ) for token_line in tokens
    ])
    valid_len = (array != vocab['<pad>']).sum(1)
    return array, valid_len

def load_data_train(batch_size, num_steps, num_examples = 600):
    """
    返回数据迭代器和词汇表

    Parameters
    ----------
    @param batch_size: int
    @param num_steps: int
        The length of the sequence of tokens in a batch
    @param num_examples: int
        The number of examples in the dataset
    @return: tuple of (data_iter, en_vocab, cn_vocab)
        - data_iter: instance of torch.utils.data.DataLoader
        - en_vocab: instance of Vocab for English
        - cn_vocab: instance of Vocab for Chinese
    """
    en_train = read_en_train()
    cn_train = read_cn_train()
    en_tokens = tokenize(en_train, num_examples)
    cn_tokens = tokenize(cn_train, num_examples)
    en_vocab = Vocab(en_tokens, min_freq=2, reserved_tokens=['<pad>', '<bos>', '<eos>'])
    cn_vocab = Vocab(cn_tokens, min_freq=2, reserved_tokens=['<pad>', '<bos>', '<eos>'])
    en_array, en_valid_len = build_array(en_tokens, en_vocab, num_steps)
    cn_array, cn_valid_len = build_array(cn_tokens, cn_vocab, num_steps)
    dataset = torch.utils.data.TensorDataset(en_array, en_valid_len, cn_array, cn_valid_len)
    data_iter = torch.utils.data.DataLoader(dataset, batch_size, shuffle=True)
    return data_iter, en_vocab, cn_vocab
