import regex as re
from collections import defaultdict

def single_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    # single process implementation:
    # 0. split corpus to exclude special_tokens, add special_tokens to vocab
    # 1. pretokenize corpus
    # 2. bpe merging until vocab size reaches vocab_size (size(special_tokens)+256 bytes+size(merged_tokens)
        # naive bpe implementation is ~O(n^2) because for each merge, we need to interate over entire token_freq to generate updated adj_freq to find next to be 
        # merged byte pair, but we can use pair_pretoken hashmap to reduce iteration-needed sequence to containing pretokens only
    # 3. output vocab and merges

    # construct vocab = dict[int, bytes]
    vocab = dict(int, bytes)
    for i in range (0,256):
        vocab[i] = bytes(i)

    for i,sp_token in enumerate(special_tokens):
        vocab[i+255] = sp_token


    # TODO to load large file we shouldn't load entire thing into RAM right, maybe yield?
    with open(input_path, "r") as f:
        content = f.read()
    # split corpus and exclude special tokens and pretokenize
    target_tokens = re.split("|".join(special_tokens), content)

    # build pretoken count table & pair_pretoken table & pair_count table
    pretoken_count = defaultdict(int)
    pair_count = defaultdict(int)
    pair_pretoken = defaultdict(set)

    for token in target_tokens:
        # byte stringize token
        btoken = token.encode("utf-8")
        pretoken_count[btoken] += 1
        for i in range(len(btoken)-1):
            curr_pair = tuple(btoken[i], btoken[i+1])
            pair_count[curr_pair] += 1
            pair_pretoken[curr_pair].add(btoken) # storing btoken or string represengint containing pretoken are nearly identical here?
    # list[tuple[bytes, bytes]]
    merged_pair_list = []
    while len(vocab) < vocab_size and len(pair_count) != 0:
        # grab curr max count pairs from pair_count, max() returns our desired alphabetic order
        to_merge_pair = max(pair_count, key=lambda k: (pair_count[k], k))
        # grab merge_pair containing pretokens
        containing_tokens = pair_pretoken[to_merge_pair]
        pair_pretoken.pop(to_merge_pair)
        merged_pair = to_merge_pair[0] + to_merge_pair[1]
        for token in containing_tokens:
            # iterate ovre token to find left/right neighbor of each to_merge_pair occurance, and to update them to 
            # x,a,b,y -> (x,a),(a,b),(b,y) -> (x,ab),(ab,y)
            for i in token: #TODO
                pair_count[left_neighbor] -= 1
                pair_count[right_neighbor] -= 1
                pair_count[to_merge_pair] -= 1
            pair_pretoken[merged_pair].add(token)

        pair_count[merged_pair] += 1
        vocab[len(vocab)] = merged_pair
        merged_pair_list.append(to_merge_pair)


    
        
        








