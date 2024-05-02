# -*- coding: utf-8 -*-

''' all transformer code adapted from PyTorch translation transformer 
    tutorial found here:
    https://pytorch.org/tutorials/beginner/translation_transformer.html
'''

import torch
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from typing import Iterable, List
from spacy.lang.en import English
import english_simple_sent_align as align
import os
import random

#added RT


# for the folder of saved models
CKPT_DIR = "./code/ckpt/dissimilar"

# for switching between different lists of sentences
TRAIN_STR = "train" # make sure that the list is also shuffled before chaging back to "train" (in sent_pairs)

SRC_LANGUAGE = "en"
TGT_LANGUAGE = "simple"

token_transform = {} # holds the tokenizers
vocab_transform = {} # holds the tokenized sentences 


"""Create source and target language tokenizer. Make sure to install the
dependencies.

``` {.sourceCode .python}
pip install -U torchdata
pip install -U spacy
python -m spacy download en_core_web_sm
python -m spacy download de_core_news_sm
```

"""
token_transform[SRC_LANGUAGE] = get_tokenizer(tokenizer=None)
token_transform[TGT_LANGUAGE] = get_tokenizer(tokenizer=None)

# helper function to yield list of tokens
def yield_tokens(data_iter: Iterable, language: str) -> List[str]:
    # creates an index
    language_index = {SRC_LANGUAGE: 0, TGT_LANGUAGE: 1}

# lets you return many things several times...
# not the same as a list
    for data_sample in data_iter:
        yield token_transform[language](data_sample[language_index[language]])

# got it. https://github.com/nicolas-ivanov/tf_seq2seq_chatbot/issues/15
# unknown, padding, bos (beginning of sentence?), end of sentence
# Define special symbols and indices
UNK_IDX, PAD_IDX, BOS_IDX, EOS_IDX = 0, 1, 2, 3
# Make sure the tokens are in order of their indices to properly insert them in vocab
special_symbols = ['<unk>', '<pad>', '<bos>', '<eos>']

train_iter = align.sent_pairs(TRAIN_STR)

for ln in [SRC_LANGUAGE, TGT_LANGUAGE]:
    # Training data Iterator
    # Create torchtext's Vocab object
    # (special first is askes if we have start and end symbols--yes)
    vocab_transform[ln] = build_vocab_from_iterator(yield_tokens(train_iter, ln),
                                                    min_freq=1,
                                                    specials=special_symbols,
                                                    special_first=True)

# not sure why this is useful yet, but okay
# Set ``UNK_IDX`` as the default index. This index is returned when the token is not found.
# If not set, it throws ``RuntimeError`` when the queried token is not found in the Vocabulary.
for ln in [SRC_LANGUAGE, TGT_LANGUAGE]:
  vocab_transform[ln].set_default_index(UNK_IDX)

"""Seq2Seq Network using Transformer
=================================

Transformer is a Seq2Seq model introduced in ["Attention is all you
need"](https://papers.nips.cc/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf)
paper for solving machine translation tasks. Below, we will create a
Seq2Seq network that uses Transformer. The network consists of three
parts. First part is the embedding layer. This layer converts tensor of
input indices into corresponding tensor of input embeddings. These
embedding are further augmented with positional encodings to provide
position information of input tokens to the model. The second part is
the actual
[Transformer](https://pytorch.org/docs/stable/generated/torch.nn.Transformer.html)
model. Finally, the output of the Transformer model is passed through
linear layer that gives unnormalized probabilities for each token in the
target language.

"""

from torch import Tensor
import torch
import torch.nn as nn
from torch.nn import Transformer
import math
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# step 1. embedding layer
# helper Module that adds positional encoding to the token embedding to introduce a notion of word order.
class PositionalEncoding(nn.Module):
    def __init__(self,
                 emb_size: int,
                 dropout: float,
                 maxlen: int = 5000):
        super(PositionalEncoding, self).__init__()
        den = torch.exp(- torch.arange(0, emb_size, 2)* math.log(10000) / emb_size)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(-2)

        self.dropout = nn.Dropout(dropout)
        self.register_buffer('pos_embedding', pos_embedding)

    def forward(self, token_embedding: Tensor):
        return self.dropout(token_embedding + self.pos_embedding[:token_embedding.size(0), :])

# look into this more- place to use our embeddings!!!!
# helper Module to convert tensor of input indices into corresponding tensor of token embeddings
class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, emb_size):
        super(TokenEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.emb_size = emb_size

    def forward(self, tokens: Tensor):
        return self.embedding(tokens.long()) * math.sqrt(self.emb_size)

# step 2. transformer
# Seq2Seq Network
class Seq2SeqTransformer(nn.Module):
    def __init__(self,
                 num_encoder_layers: int,
                 num_decoder_layers: int,
                 emb_size: int,
                 nhead: int,
                 src_vocab_size: int,
                 tgt_vocab_size: int,
                 dim_feedforward: int = 512,
                 dropout: float = 0.1):
        super(Seq2SeqTransformer, self).__init__()
        self.transformer = Transformer(d_model=emb_size,
                                       nhead=nhead,
                                       num_encoder_layers=num_encoder_layers,
                                       num_decoder_layers=num_decoder_layers,
                                       dim_feedforward=dim_feedforward,
                                       dropout=dropout)
        self.generator = nn.Linear(emb_size, tgt_vocab_size)
        self.src_tok_emb = TokenEmbedding(src_vocab_size, emb_size)
        self.tgt_tok_emb = TokenEmbedding(tgt_vocab_size, emb_size)
        self.positional_encoding = PositionalEncoding(
            emb_size, dropout=dropout)

    def forward(self,
                src: Tensor,
                trg: Tensor,
                src_mask: Tensor,
                tgt_mask: Tensor,
                src_padding_mask: Tensor,
                tgt_padding_mask: Tensor,
                memory_key_padding_mask: Tensor):
        src_emb = self.positional_encoding(self.src_tok_emb(src))
        tgt_emb = self.positional_encoding(self.tgt_tok_emb(trg))
        outs = self.transformer(src_emb, tgt_emb, src_mask, tgt_mask, None,
                                src_padding_mask, tgt_padding_mask, memory_key_padding_mask)
        return self.generator(outs)

    def encode(self, src: Tensor, src_mask: Tensor):
        return self.transformer.encoder(self.positional_encoding(
                            self.src_tok_emb(src)), src_mask)

    def decode(self, tgt: Tensor, memory: Tensor, tgt_mask: Tensor):
        return self.transformer.decoder(self.positional_encoding(
                          self.tgt_tok_emb(tgt)), memory,
                          tgt_mask)

"""During training, we need a subsequent word mask that will prevent the
model from looking into the future words when making predictions. We
will also need masks to hide source and target padding tokens. Below,
let\'s define a function that will take care of both.

"""

# masking!!! hides future words?
# this gets used in create mask below
def generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones((sz, sz), device=DEVICE)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask

# masking!!! hides source and target padding tokens
def create_mask(src, tgt):
    src_seq_len = src.shape[0]
    tgt_seq_len = tgt.shape[0]

    tgt_mask = generate_square_subsequent_mask(tgt_seq_len)
    src_mask = torch.zeros((src_seq_len, src_seq_len),device=DEVICE).type(torch.bool)

    src_padding_mask = (src == PAD_IDX).transpose(0, 1)
    tgt_padding_mask = (tgt == PAD_IDX).transpose(0, 1)

    # returns four masks!!!!
    return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask

"""Let\'s now define the parameters of our model and instantiate the same.
Below, we also define our loss function which is the cross-entropy loss
and the optimizer used for training.

"""

''' make_or_restore_model is adapted from from Vassar CMPU 366 
    Assignment 5, which was adapted from an assignment developed 
    by Carolyn Anderson, Wellesley College.
'''
def make_or_restore_model():
    ''' recreates model based on saved versions in case of not being able to complete
        training in one go
    '''
    transformer = Seq2SeqTransformer(NUM_ENCODER_LAYERS, NUM_DECODER_LAYERS, EMB_SIZE,
                                    NHEAD, SRC_VOCAB_SIZE, TGT_VOCAB_SIZE, FFN_HID_DIM)

    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    transformer = transformer.to(DEVICE)

    optimizer = torch.optim.Adam(transformer.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)

    checkpoints = [
        CKPT_DIR + "/" + name
        for name in os.listdir(CKPT_DIR)
        if name[-1] == "t"
    ]

    if checkpoints:
        latest_checkpoint = max(checkpoints, key=os.path.getctime)

        print("Restoring from", latest_checkpoint)
        ckpt = torch.load(latest_checkpoint)
        transformer.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        epoch = ckpt["epoch"]
        return transformer, optimizer, epoch + 1
    else:
        print("Creating a new model")
        return transformer, optimizer, 1

torch.manual_seed(0)

# RC is source and TGT is target
SRC_VOCAB_SIZE = len(vocab_transform[SRC_LANGUAGE])
TGT_VOCAB_SIZE = len(vocab_transform[TGT_LANGUAGE])

# i will assume these numbers are fine and have no particular stuff we need to change
# exCEPT WE NEED TO SHRINK SO BAD
EMB_SIZE = 512
NHEAD = 8
FFN_HID_DIM = 512
BATCH_SIZE = 12
NUM_ENCODER_LAYERS = 3
NUM_DECODER_LAYERS = 3

loss_fn = torch.nn.CrossEntropyLoss(ignore_index=PAD_IDX)

transformer, optimizer, next_epoch = make_or_restore_model()

"""Collation
=========

As seen in the `Data Sourcing and Processing` section, our data iterator
yields a pair of raw strings. We need to convert these string pairs into
the batched tensors that can be processed by our `Seq2Seq` network
defined previously. Below we define our collate function that converts a
batch of raw strings into batch tensors that can be fed directly into
our model.

"""

# goal: convert strings to batched tensors to be useable by seq2seq

from torch.nn.utils.rnn import pad_sequence

# helper function to club together sequential operations
def sequential_transforms(*transforms):
    def func(txt_input):
        for transform in transforms:
            txt_input = transform(txt_input)
        return txt_input
    return func

# function to add BOS/EOS and create tensor for input sequence indices
def tensor_transform(token_ids: List[int]):
    # torch.cat 'concatenates the given sequence of seq tensors in the given dimension'.
    # they must be the same size or a size 1-D one with size 0!
    return torch.cat((torch.tensor([BOS_IDX]),
                      torch.tensor(token_ids),
                      torch.tensor([EOS_IDX])))

# ``src`` and ``tgt`` language text transforms to convert raw strings into tensors indices
text_transform = {}
for ln in [SRC_LANGUAGE, TGT_LANGUAGE]:
    # okay. token_transform is from the BEGINNING and is the tokenized stuff okay
    text_transform[ln] = sequential_transforms(token_transform[ln], #Tokenization
                                               # makes it into a Vocab Object as stated earlier
                                               vocab_transform[ln], #Numericalization
                                               tensor_transform) # Add BOS/EOS and create tensor


# function to collate data samples into batch tensors
def collate_fn(batch):
    src_batch, tgt_batch = [], []
    for src_sample, tgt_sample in batch:
        src_batch.append(text_transform[SRC_LANGUAGE](src_sample.rstrip("\n")))
        tgt_batch.append(text_transform[TGT_LANGUAGE](tgt_sample.rstrip("\n")))

    src_batch = pad_sequence(src_batch, padding_value=PAD_IDX)
    tgt_batch = pad_sequence(tgt_batch, padding_value=PAD_IDX)
    return src_batch, tgt_batch

"""Let\'s define training and evaluation loop that will be called for each
epoch.

"""

from torch.utils.data import DataLoader

def train_epoch(model, optimizer):
    model.train()
    losses = 0
    train_dataloader = DataLoader(train_iter, batch_size=BATCH_SIZE, collate_fn=collate_fn)

    for src, tgt in train_dataloader:
        src = src.to(DEVICE)
        tgt = tgt.to(DEVICE)

        tgt_input = tgt[:-1, :]

        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input)

        logits = model(src, tgt_input, src_mask, tgt_mask,src_padding_mask, tgt_padding_mask, src_padding_mask)

        optimizer.zero_grad()

        tgt_out = tgt[1:, :]
        loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
        loss.backward()

        optimizer.step()
        losses += loss.item()
    
    return losses / len(list(train_dataloader))


def evaluate(model):
    model.eval()
    losses = 0

    val_iter = align.sent_pairs("valid")
    val_dataloader = DataLoader(val_iter, batch_size=BATCH_SIZE, collate_fn=collate_fn)

    for src, tgt in val_dataloader:
        src = src.to(DEVICE)
        tgt = tgt.to(DEVICE)

        tgt_input = tgt[:-1, :]

        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input)

        logits = model(src, tgt_input, src_mask, tgt_mask,src_padding_mask, tgt_padding_mask, src_padding_mask)

        tgt_out = tgt[1:, :]
        loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
        losses += loss.item()

    return losses / len(list(val_dataloader))

"""Now we have all the ingredients to train our model. Let\'s do it!

"""

from timeit import default_timer as timer
NUM_EPOCHS = 5

for epoch in range(next_epoch, NUM_EPOCHS+1):
    start_time = timer()
    train_loss = train_epoch(transformer, optimizer)
    end_time = timer()
    val_loss = evaluate(transformer)
    print((f"Epoch: {epoch}, Train loss: {train_loss:.3f}, Val loss: {val_loss:.3f}, "f"Epoch time = {(end_time - start_time):.3f}s"))
    
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": transformer.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": train_loss,
        },
        f"{CKPT_DIR}/ckpt_{epoch}.pt",
    )


# function to generate output sequence using an algorithm to
# consider top 3 options for the next word
def one_of_top_3_decode(model, src, src_mask, max_len, start_symbol):
    src = src.to(DEVICE)
    src_mask = src_mask.to(DEVICE)

    memory = model.encode(src, src_mask)
    ys = torch.ones(1, 1).fill_(start_symbol).type(torch.long).to(DEVICE)
    for i in range(max_len-1):
        memory = memory.to(DEVICE)
        tgt_mask = (generate_square_subsequent_mask(ys.size(0))
                    .type(torch.bool)).to(DEVICE)
        out = model.decode(ys, memory, tgt_mask)
        out = out.transpose(0, 1)
        prob = model.generator(out[:, -1])
        #pick random from top 3
        _, next_word_indices = torch.topk(prob, 3, dim=1, largest=True)
        try:
            rand_num = random.randint(0, 3)
            next_word_index = next_word_indices.data[rand_num]
            next_word = next_word_index.item()
        except:
            _, next_word = torch.max(prob, dim=1)
            next_word = next_word.item()
                
        #do we want to keep something of the probability
        #rating to hold onto for the reranking

        ys = torch.cat([ys,
                        torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=0)
        if next_word == EOS_IDX:
            break
    return ys


# https://www.scaler.com/topics/levenshtein-distance-python/#
def levenshteinDistance(A, B):
    N, M = len(A), len(B)
    # Create an array of size NxM
    dp = [[0 for i in range(M + 1)] for j in range(N + 1)]

    # Base Case: When N = 0
    for j in range(M + 1):
        dp[0][j] = j
    # Base Case: When M = 0
    for i in range(N + 1):
        dp[i][0] = i
    # Transitions
    for i in range(1, N + 1):
        for j in range(1, M + 1):
            if A[i - 1] == B[j - 1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(
                    dp[i-1][j], # Insertion
                    dp[i][j-1], # Deletion
                    dp[i-1][j-1] # Replacement
                )

    return dp[N][M]


# actual function to translate input sentence into target language
def translate(model: torch.nn.Module, src_sentence: str):
    model.eval()
    src = text_transform[SRC_LANGUAGE](src_sentence).view(-1, 1)
    num_tokens = src.shape[0]
    src_mask = (torch.zeros(num_tokens, num_tokens)).type(torch.bool)
    tgt_tokens = one_of_top_3_decode(
        model,  src, src_mask, max_len=num_tokens + 5, start_symbol=BOS_IDX).flatten()
    return " ".join(vocab_transform[TGT_LANGUAGE].lookup_tokens(list(tgt_tokens.cpu().numpy()))).replace("<bos>", "").replace("<eos>", "")

def run_ten_times(to_translate: str):
    poss_list = []
    for i in range(0, 10):
        #do not currently have access to translate so can only hope this works rn
        #check later
        poss_list.append(translate(transformer, to_translate))
    return poss_list
        

# takes in list and reranks by levenshtein distance
# returns highest lev distance
def rerank_and_top(potential_list, original):
    potential_lev = []
    for i in range(0, len(potential_list)):
        potential_lev.append((potential_list[i], levenshteinDistance(potential_list[i], original)))

    #sort by value
    sort_poss = sorted(potential_lev, key=lambda x: x[1], reverse=True)
    return sort_poss[0][0]

###RT add
##to_trans = "If music be the food of love , play on ."
##translated_v = translate(transformer, to_trans)
### I think the idea of the reference here might be a little funky
### because like we probably only have one version we would call correct?
### this should totally end as a helper function
##bleu_score = sentence_bleu([to_trans.split()], translated_v.split())
##print('BLEU score -> {}'.format(bleu_score)) 


poss_list = run_ten_times("If music be the food of love , play on .")
top = rerank_and_top(poss_list, "If music be the food of love , play on .")
print(top)

"""References
==========
1.  PyTorch translation transformer tutorial 
    <https://pytorch.org/tutorials/beginner/translation_transformer.html>
2.  Vassar CMPU 366 Assignment 5, adapted from an assignment developed by 
    Carolyn Anderson, Wellesley College.
3.  Attention is all you need paper.
    <https://papers.nips.cc/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf>
4.  The annotated transformer.
    <https://nlp.seas.harvard.edu/2018/04/03/attention.html#positional-encoding>

"""
