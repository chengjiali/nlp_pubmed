
def gen_word2vec_train_example(sent, context_size=2, pad_token=0):
    '''
    Generate Word2Vec training example, i.e. left + right context -> target.
    [0, 0, w_2, w_3] -> w_1, ... [w_n-2, w_n-1, 0, 0] -> w_n
    '''
    
    # Pad the sentence with <S> token
    padded_sent = [pad_token] * context_size + sent + [pad_token] * context_size

    # Generate training examples
    for i in range(context_size, len(sent) + context_size):
        context = padded_sent[i - N:i] + padded_sent[i + 1:i + N + 1]
        target = [padded_sent[i]]
        
        yield context, target

def gen_lm_train_example(sent, context_size=2, pad_token=0):
    '''
    Generate LM training example, i.e. context -> target.
    [0, 0] -> w_1, ... [w_n-1, w_n] -> 0
    '''
    
    # Pad the sentence with [S] token
    padded_sent = sent + [pad_token]

    # Generate training examples
    context = [pad_token] * context_size
    for target in padded_sent:
        yield context, target
        context = context[1:] + [target]