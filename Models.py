import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class NGramLM(nn.Module):
    '''
    N-Gram language model.
    Given a sequence of words, predict the next probability of next word
    using P(w_i | w_i-1, w_i-2, ... , w_i-n+1).
    Context are treated as a bag of words.
    '''
    def __init__(self, vocab_size, embed_dim, context_size, hidden_size):
        super(NGramLM, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.fnn = nn.Sequential(
            nn.Linear(context_size * embed_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, vocab_size),
        )
        
    def forward(self, inputs):
        embed = self.embed(inputs)                 # [batch_size, context_size, embed_dim]
        embed = embed.view((embed.size(0), -1))    # [batch_size, context_size * embed_dim]
        logits = self.fnn(embed)
        probas = F.log_softmax(logits)
        
        # nn.CrossEntropyLoss = nn.LogSoftmax() + nn.NLLLoss
        # https://pytorch.org/docs/stable/nn.html#torch.nn.CrossEntropyLoss
        # Return logits if using nn.CrossEntropyLoss, probas if using nn.NLLLoss
        # Log softmax > softmax in terms of performance
        return logits, probs
    

class Word2VecCBOW(nn.Module):
    '''
    Word2Vec Continuous Bag-of-Words.
    Given a target w_i and N context words on both sides of w_i,
    minimize -log(P(w_i|C)) = -log(softmax(W * sum(w_c) + b))),
    where w_c is the vector of context word.
    '''
    def __init__(self, vocab_size, embed_dim):
        super(Word2VecCBOW, self).__init__()

        self.embed = nn.Embedding(vocab_size, embed_dim)
        nn.init.uniform_(self.embed.weight, -0.25, 0.25)
        
        """ projection layer for taking softmax over vocabulary words"""
        self.projection = torch.nn.Linear(embed_dim, )
        nn.init.uniform_(self.projection.weight, -0.25, 0.25)

    def forward(self, words):
        embed = self.embedding(words)
        embed_sum = torch.sum(embed, dim=0)  # size(emb_sum) = emb_size
        embed_sum = embed_sum.view(1, -1)  # size(emb_sum) = 1 x emb_size
        out = self.projection(emb_sum)  # size(out) = 1 x nwords
        return out

    
class Word2VecSkipGram(nn.Module):
    '''
    Word2Vec Skip Gram.
    Given a target w_i and N context words on both sides of w_i,
    minimize -log(P(C|w_i)) = -log(softmax(W * sum(w_c) + b))),
    where w_c is the vector of context word.
    '''
    def __init__(self, vocab_size, embed_dim):
        super(Word2VecSkipGram, self).__init__()

        self.embed = nn.Embedding(vocab_size, embed_dim)
        nn.init.uniform_(self.embed.weight, -0.25, 0.25)
        
        """ projection layer for taking softmax over vocabulary words"""
        self.projection = torch.nn.Linear(embed_dim, )
        nn.init.uniform_(self.projection.weight, -0.25, 0.25)

    def forward(self, words):
        emb = self.embedding(words)
        emb_sum = torch.sum(emb, dim=0)  # size(emb_sum) = emb_size
        emb_sum = emb_sum.view(1, -1)  # size(emb_sum) = 1 x emb_size
        out = self.projection(emb_sum)  # size(out) = 1 x nwords
        return out