import math
import torch
from torch import nn
import matplotlib.pyplot as plt

def show_attention(
        attention,
        xlabel = 'Keys',
        ylabel = 'Queries',
        title = 'Attention weights',
        figsize=(5, 5),
        cmap = 'Reds'
    ):
    """
    画出注意力权重图

    Parameters
    ----------
    @param attention: torch.Tensor
        attention weights, shape (m, n)
    """

    fig = plt.figure(figsize = figsize)

    pcm = plt.imshow(
        attention.detach().numpy(), 
        cmap = cmap
    )

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    fig.colorbar(pcm, shrink=0.7)
    plt.show()

def masked_softmax(X, valid_lens):  #@save
    """
    Perform softmax operation by masking elements on the last axis.
    
    Parameters
    ----------
    X : torch.Tensor
        3D tensor whose last axis has values to be masked.
    valid_lens : torch.Tensor
        1D or 2D tensor consisting of valid lengths of sequences.
    """
    def _sequence_mask(
            X, 
            valid_len, 
            value=0
        ):
        maxlen = X.size(1)
        mask = torch.arange(
            (maxlen),
            dtype=torch.float32,
            device=X.device
        )[None, :] < valid_len[:, None]
        X[~mask] = value
        return X

    if valid_lens is None:
        return nn.functional.softmax(X, dim=-1)
    else:
        shape = X.shape
        if valid_lens.dim() == 1:
            valid_lens = torch.repeat_interleave(
                valid_lens, shape[1]
            )
        else:
            valid_lens = valid_lens.reshape(-1)
        # On the last axis, replace masked elements with a very large negative
        # value, whose exponentiation outputs 0
        X = _sequence_mask(
            X.reshape(-1, shape[-1]), 
            valid_lens, 
            value=-1e6
        )
        return nn.functional.softmax(
            X.reshape(shape), dim=-1
        )

class AdditiveAttention(nn.Module):  #@save
    """Additive attention."""
    def __init__(self, num_hiddens, dropout, **kwargs):
        super(AdditiveAttention, self).__init__(**kwargs)
        self.W_k = nn.LazyLinear(num_hiddens, bias=False)
        self.W_q = nn.LazyLinear(num_hiddens, bias=False)
        self.w_v = nn.LazyLinear(1, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, queries, keys, values, valid_lens):
        queries, keys = self.W_q(queries), self.W_k(keys)
        # After dimension expansion, shape of queries: (batch_size, no. of
        # queries, 1, num_hiddens) and shape of keys: (batch_size, 1, no. of
        # key-value pairs, num_hiddens). Sum them up with broadcasting
        features = queries.unsqueeze(2) + keys.unsqueeze(1)
        features = torch.tanh(features)
        # There is only one output of self.w_v, so we remove the last
        # one-dimensional entry from the shape. Shape of scores: (batch_size,
        # no. of queries, no. of key-value pairs)
        scores = self.w_v(features).squeeze(-1)
        self.attention_weights = masked_softmax(
            scores, valid_lens
        )
        # Shape of values: (batch_size, no. of key-value pairs, value
        # dimension)
        return torch.bmm(
            self.dropout(self.attention_weights), 
            values
        )
    
class DotProductAttention(nn.Module):  #@save
    def __init__(self, dropout):
        """Scaled dot product attention."""
        super().__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(self, queries, keys, values, valid_lens=None):
        """
        Parameters
        ----------
        @param queries: torch.Tensor
            Shape (batch_size, no. of queries, d)
        @param keys: torch.Tensor
            Shape (batch_size, no. of key-value pairs, d)
        @param values: torch.Tensor
            Shape (batch_size, no. of key-value pairs, value dimension)
        @param valid_lens: torch.Tensor
            Either shape (batch_size, ) or (batch_size, no. of queries).
        @return output: torch.Tensor
            Shape (batch_size, no. of queries, value dimension)
        """
        d = queries.shape[-1]
        # Swap the last two dimensions of keys with keys.transpose(1, 2)
        scores = torch.bmm(
            queries, 
            keys.transpose(1, 2)
        ) / math.sqrt(d)
        self.attention_weights = masked_softmax(
            scores, 
            valid_lens
        )
        return torch.bmm(
            self.dropout(self.attention_weights), 
            values
        )
    
class MultiHeadAttention(nn.Module):  #@save
    def __init__(self, num_hiddens, num_heads, dropout, bias=False, **kwargs):
        """
        Multi-head attention.

        Parameters
        ----------
        @param num_hiddens: int
            hidden size
        @param num_heads: int
            number of heads
        @param dropout: float
            dropout rate
        @param bias: bool
            whether to use bias
        @param kwargs: dict
            other parameters
        """
        super().__init__()
        self.num_heads = num_heads
        assert num_hiddens % num_heads == 0
        self.attention = DotProductAttention(dropout)
        self.W_q = nn.LazyLinear(num_hiddens, bias=bias)
        self.W_k = nn.LazyLinear(num_hiddens, bias=bias)
        self.W_v = nn.LazyLinear(num_hiddens, bias=bias)
        self.W_o = nn.LazyLinear(num_hiddens, bias=bias)

    def forward(self, queries, keys, values, valid_lens):
        """
        Parameters
        ----------
        @param queries: torch.Tensor
            Shape (batch_size, no. of queries, num_hiddens)
        @param keys: torch.Tensor
            Shape (batch_size, no. of key-value pairs, num_hiddens)
        @param values: torch.Tensor
            Shape (batch_size, no. of key-value pairs, num_hiddens)
        @param valid_lens: torch.Tensor
            Either shape (batch_size, ) or (batch_size, no. of queries).
        @return output: torch.Tensor
            Shape (batch_size, no. of queries, num_hiddens)
        """
        # After transposing, 
        # shape of output queries, keys, or values:
        # (
        #   batch_size * num_heads, 
        #   no. of queries or key-value pairs,
        #   num_hiddens / num_heads
        # )
        queries = self.transpose_qkv(self.W_q(queries))
        keys = self.transpose_qkv(self.W_k(keys))
        values = self.transpose_qkv(self.W_v(values))

        if valid_lens is not None:
            # On axis 0, copy the first item (scalar or vector) for num_heads
            # times, then copy the next item, and so on
            valid_lens = torch.repeat_interleave(
                valid_lens, repeats=self.num_heads, dim=0
            )

        # Shape of output: 
        # (
        #   batch_size * num_heads, 
        #   no. of queries,
        #   num_hiddens / num_heads
        # )
        output = self.attention(
            queries, 
            keys, 
            values, 
            valid_lens
        )

        # Shape of output_concat: 
        # (
        #   batch_size, 
        #   no. of queries, 
        #   num_hiddens
        # )
        output_concat = self.transpose_output(output)
        return self.W_o(output_concat)
    
    def transpose_qkv(self, X):
        """
        Transposition for parallel computation of multiple attention heads.
        
        Parameters
        ----------
        @param X: torch.Tensor
            Shape 
            (
                batch_size, 
                no. of queries or key-value pairs, 
                num_hiddens
            ).
        @return X: torch.Tensor
            Shape 
            (
                batch_size * num_heads, 
                no. of queries or key-value pairs, 
                num_hiddens / num_heads
            )
        """
        # Shape of input X: 
        # (
        #   batch_size, 
        #   no. of queries or key-value pairs,
        #   num_hiddens
        # ). 
        # Shape of output X: 
        # (
        #   batch_size, 
        #   no. of queries or key-value pairs, 
        #   num_heads, 
        #   num_hiddens / num_heads
        # )
        X = X.reshape(X.shape[0], X.shape[1], self.num_heads, -1)

        # Shape of output X: 
        # (
        #   batch_size, 
        #   num_heads, 
        #   no. of queries or key-value pairs, 
        #   num_hiddens / num_heads
        # )
        X = X.permute(0, 2, 1, 3)

        # Shape of output: 
        # (
        #   batch_size * num_heads, 
        #   no. of queries or key-value pairs, 
        #   num_hiddens / num_heads
        # )
        return X.reshape(-1, X.shape[2], X.shape[3])

    def transpose_output(self, X):
        """
        Reverse the operation of transpose_qkv.
        
        Parameters
        ----------
        @param X: torch.Tensor
            Shape 
            (
                batch_size * num_heads, 
                no. of queries,
                num_hiddens / num_heads
            ).
        @return X: torch.Tensor
            Shape 
            (
                batch_size, 
                no. of queries, 
                num_hiddens
            )
        """
        # Shape of input X:
        # (
        #   batch_size * num_heads,
        #   no. of queries,
        #   num_hiddens / num_heads
        # )
        # Shape of output X:
        # (
        #   batch_size,
        #   num_heads,
        #   no. of queries,
        #   num_hiddens / num_heads
        # )
        X = X.reshape(-1, self.num_heads, X.shape[1], X.shape[2])

        # Shape of output X:
        # (
        #   batch_size,
        #   no. of queries,
        #   num_heads,
        #   num_hiddens / num_heads
        # )
        X = X.permute(0, 2, 1, 3)

        # Shape of output X:
        # (
        #   batch_size,
        #   no. of queries,
        #   num_hiddens
        # )
        return X.reshape(X.shape[0], X.shape[1], -1)

class PositionalEncoding(nn.Module):  #@save
    def __init__(self, num_hiddens, dropout, max_len=1000):
        """
        Positional encoding.
        
        Parameters
        ----------
        @param num_hiddens: int
            hidden size
        @param dropout: float
            dropout rate
        @param max_len: int
            maximum length
        """
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        # Create a long enough P
        self.P = torch.zeros((1, max_len, num_hiddens))
        X = torch.arange(
            max_len, 
            dtype=torch.float32
        ).reshape(-1, 1) / torch.pow(
            10000, torch.arange(
                0, num_hiddens, 2, dtype=torch.float32
            ) / num_hiddens
        )
        self.P[:, :, 0::2] = torch.sin(X)
        self.P[:, :, 1::2] = torch.cos(X)

    def forward(self, X):
        """
        Parameters
        ----------
        @param X : torch.Tensor
            Shape (batch_size, num_steps, num_hiddens)
        @return torch.Tensor
            Shape (batch_size, num_steps, num_hiddens)
        """
        X = X + self.P[:, :X.shape[1], :].to(X.device)
        return self.dropout(X)
    

class PositionWiseFFN(nn.Module):  #@save
    
    def __init__(self, ffn_num_hiddens, ffn_num_outputs):
        """
        The positionwise feed-forward network.
        
        Parameters
        ----------
        @param ffn_num_hiddens: int
            hidden size
        @param ffn_num_outputs: int
            output size
        """
        super().__init__()
        self.dense1 = nn.LazyLinear(ffn_num_hiddens)
        self.relu = nn.ReLU()
        self.dense2 = nn.LazyLinear(ffn_num_outputs)

    def forward(self, X):
        """ 
        Parameters
        ----------
        @param X : torch.Tensor
            Shape (batch_size, seq_len, num_hiddens)
        @return out : torch.Tensor
            Shape (batch_size, seq_len, ffn_num_outputs)
        """
        return self.dense2(self.relu(self.dense1(X)))
    

class AddNorm(nn.Module):  #@save
    def __init__(self, norm_shape, dropout):
        """
        The residual connection followed by layer normalization.
        
        Parameters
        ----------
        @param norm_shape: int
            shape of normalization
        @param dropout: float
            dropout rate
        """
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(norm_shape) # for num_hiddens

    def forward(self, X, Y):
        """
        Parameters
        ----------
        @param X : torch.Tensor
            Shape (batch_size, seq_len, num_hiddens)
        @param Y : torch.Tensor
            Shape (batch_size, seq_len, num_hiddens)
        @return out : torch.Tensor
            Shape (batch_size, seq_len, num_hiddens)
        """
        return self.ln(self.dropout(Y) + X)
    

class TransformerEncoderBlock(nn.Module):  #@save
    def __init__(self, num_hiddens, ffn_num_hiddens, num_heads, dropout,
                 use_bias=False):
        """
        The Transformer encoder block.
        
        Parameters
        ----------
        @param num_hiddens: int
            hidden size
        @param ffn_num_hiddens: int
            hidden size of feed-forward network
        @param num_heads: int
            number of heads
        @param dropout: float
            dropout rate
        @param use_bias: bool
            whether to use bias
        """
        super().__init__()
        self.attention = MultiHeadAttention(
            num_hiddens, num_heads,
            dropout, use_bias
        )
        self.addnorm1 = AddNorm(num_hiddens, dropout)
        self.ffn = PositionWiseFFN(ffn_num_hiddens, num_hiddens)
        self.addnorm2 = AddNorm(num_hiddens, dropout)

    def forward(self, X, valid_lens):
        """
        Parameters
        ----------
        @param X : torch.Tensor
            Shape (batch_size, seq_len, num_hiddens)
        @param valid_lens : torch.Tensor
            Shape (batch_size, )
        @return out : torch.Tensor
            Shape (batch_size, seq_len, num_hiddens)
        """
        Y = self.addnorm1(X, self.attention(X, X, X, valid_lens))
        return self.addnorm2(Y, self.ffn(Y))
    

class TransformerEncoder(nn.Module):  #@save
    def __init__(self, vocab_size, num_hiddens, ffn_num_hiddens,
                 num_heads, num_blks, dropout, use_bias=False):
        """
        The Transformer encoder.
        
        Parameters
        ----------
        @param vocab_size : int
        @param num_hiddens : int
        @param ffn_num_hiddens : int
        @param num_heads : int
        @param num_blks : int
        @param dropout : float
        @param use_bias : bool
        """
        super().__init__()
        self.num_hiddens = num_hiddens
        self.embedding = nn.Embedding(vocab_size, num_hiddens)
        self.pos_encoding = PositionalEncoding(num_hiddens, dropout)
        self.blks = nn.Sequential()
        for i in range(num_blks):
            self.blks.add_module(
                "block"+str(i), 
                TransformerEncoderBlock(
                    num_hiddens, 
                    ffn_num_hiddens, 
                    num_heads, 
                    dropout, 
                    use_bias
                )
            )

    def forward(self, X, valid_lens):
        # Since positional encoding values are between -1 and 1, the embedding
        # values are multiplied by the square root of the embedding dimension
        # to rescale before they are summed up
        X = self.pos_encoding(self.embedding(X) * math.sqrt(self.num_hiddens))
        self.attention_weights = [None] * len(self.blks)
        for i, blk in enumerate(self.blks):
            X = blk(X, valid_lens)
            self.attention_weights[i] = blk.attention.attention.attention_weights
        return X
    

class TransformerDecoderBlock(nn.Module):
    def __init__(self, num_hiddens, ffn_num_hiddens, num_heads, dropout, i):
        """
        The i-th block in the Transformer decoder

        Parameters
        ----------
        @param num_hiddens : int
        @param ffn_num_hiddens : int
        @param num_heads : int
        @param dropout : float
        @param i : int
            The index of the block in the decoder
        """
        super().__init__()
        self.i = i
        self.attention1 = MultiHeadAttention(
            num_hiddens, num_heads, dropout
        )
        self.addnorm1 = AddNorm(num_hiddens, dropout)
        self.attention2 = MultiHeadAttention(
            num_hiddens, num_heads, dropout
        )
        self.addnorm2 = AddNorm(num_hiddens, dropout)
        self.ffn = PositionWiseFFN(ffn_num_hiddens, num_hiddens)
        self.addnorm3 = AddNorm(num_hiddens, dropout)

    def forward(self, X, state):
        """"
        Parameters
        ----------
        @param X : torch.Tensor
            Shape (batch_size, seq_len, num_hiddens)
        @param state : list
            the first element is the encoder output, 
            the second element is the encoder valid length, 
            The third element: decoding state (contains representations of
            outputs in the previous time step)
        """
        enc_outputs, enc_valid_lens = state[0], state[1]
        # During training, all the tokens of any output sequence are processed
        # at the same time, so state[2][self.i] is None as initialized. When
        # decoding any output sequence token by token during prediction,
        # state[2][self.i] contains representations of the decoded output at
        # the i-th block up to the current time step
        if state[2][self.i] is None:
            key_values = X
        else:
            key_values = torch.cat((state[2][self.i], X), dim=1)
        state[2][self.i] = key_values
        if self.training:
            batch_size, num_steps, _ = X.shape
            # Shape of dec_valid_lens: (batch_size, num_steps), where every
            # row is [1, 2, ..., num_steps]
            dec_valid_lens = torch.arange(
                1, num_steps + 1, device=X.device
            ).repeat(batch_size, 1)
        else:
            dec_valid_lens = None

        # Self-attention
        X2 = self.attention1(X, key_values, key_values, dec_valid_lens)
        Y = self.addnorm1(X, X2)

        # Encoder-decoder attention. Shape of enc_outputs:
        # (batch_size, num_steps, num_hiddens)
        Y2 = self.attention2(Y, enc_outputs, enc_outputs, enc_valid_lens)
        Z = self.addnorm2(Y, Y2)
        return self.addnorm3(Z, self.ffn(Z)), state
    

class TransformerDecoder(nn.Module):
    def __init__(self, vocab_size, num_hiddens, ffn_num_hiddens, num_heads,
                 num_blks, dropout):
        """
        The Transformer decoder.

        Parameters
        ----------
        @param vocab_size : int
        @param num_hiddens : int
        @param ffn_num_hiddens : int
        @param num_heads : int
        @param num_blks : int
        @param dropout : float
        """
        super().__init__()
        self.num_hiddens = num_hiddens
        self.num_blks = num_blks
        self.embedding = nn.Embedding(vocab_size, num_hiddens)
        self.pos_encoding = PositionalEncoding(num_hiddens, dropout)
        self.blks = nn.Sequential()
        for i in range(num_blks):
            self.blks.add_module(
                "block"+str(i), 
                TransformerDecoderBlock(
                    num_hiddens, ffn_num_hiddens, 
                    num_heads, dropout, i
                )
            )
        self.dense = nn.LazyLinear(vocab_size)

    def init_state(self, enc_outputs, enc_valid_lens):
        """
        Initialize the decoder state.

        Parameters
        ----------
        @param enc_outputs : torch.Tensor
            Shape (batch_size, seq_len, num_hiddens)
        @param enc_valid_lens : torch.Tensor
            Shape (batch_size, )
        @return state : list
            The first element: enc_outputs,
            The second element: enc_valid_lens,
            The third element: list 
                decoding state (contains representations of outputs in the previous time step)
        """
        return [enc_outputs, enc_valid_lens, [None] * self.num_blks]

    def forward(self, X, state):
        """
        Parameters
        ----------
        @param X : torch.Tensor
            Shape (batch_size, seq_len)
        @param state : list
            The first element: enc_outputs, 
            The second element: enc_valid_lens, 
            The third element: list 
                decoding state (contains representations of outputs in the previous time step)
        """
        X = self.pos_encoding(self.embedding(X) * math.sqrt(self.num_hiddens))
        self._attention_weights = [[None] * len(self.blks) for _ in range (2)]
        for i, blk in enumerate(self.blks):
            X, state = blk(X, state)
            # Decoder self-attention weights
            self._attention_weights[0][i] = blk.attention1.attention.attention_weights
            # Encoder-decoder attention weights
            self._attention_weights[1][i] = blk.attention2.attention.attention_weights
        return self.dense(X), state

    @property
    def attention_weights(self):
        return self._attention_weights
    

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
      
