import torch
from torchtext.data.utils import get_tokenizer
import random

#FOR NOW DOES NOT WORK!!! DOES NOT HAVE ACCESS TO THE THINGS IT NEEDS

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
        _, next_word_indices = torch.topk(prob, 3, dim=1, largest=true)
        rand_num = random.randint(0, 2)
        next_word = next_word_indices[randnum]
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
        potential_lev.append(potential_list[i], levenshteinDistance(potential_list[i], original))

    #sort by value
    sort_poss = sorted(possible, key=lambda x: x[1], reverse=True)
    return sort_poss[0][0]
    


poss_list = run_ten_times(input_text)
top = rerank_and_top(poss_list)


