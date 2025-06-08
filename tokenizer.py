from collections import defaultdict
import json
from dataclasses import dataclass

@dataclass
class TokenizationOutput:
    tokens: list
    ids: list

class BPETokenizer:
    def __init__(self, vocab_size = 100000, merge_count = 100000, vocab_file = None):
        self.vocab_size = vocab_size
        self.merge_count = merge_count
        self.vocab = self.load_vocab(vocab_file) if vocab_file is not None else []
        self.words = []
        self.pair_count = defaultdict(int) # keep track of adjacent pair counts,to select max pair for merging

    def _preprocess(self, text):

        # split into words and add Ġ to the start of each word
        words = text.strip().split()
        return ['Ġ' + word for word in words] # ex: Ġhello, Ġworld
    
    def _post_process(self, tokens): # remove Ġ during output 
        return [token.replace('Ġ', '') for token in tokens]

    def _initialize(self, corpus):

        # create a vocab with all characters in corpus
        # create list of words split into chars [['h','e','l','l','o'], [..], ..]
        for sentence in corpus:
            words = self._preprocess(sentence)
            for word in words:
                word_list = list(word)
                self.words.append(word_list)
                for token in word:
                    if token not in self.vocab:
                        self.vocab.append(token)
                    self.pair_count[token] += 1
        
        print(f'[info] Vocabulary initialized')

    def _get_most_freq_adj_pair(self):

        pair_counts = defaultdict(int)

        for word in self.words:
            for i in range(len(word) - 1):
                pair = (word[i], word[i+1])
                pair_counts[pair] += 1

        if not pair_counts:
            return None

        pair = max(pair_counts, key = pair_counts.get)

        if pair_counts[pair] > 0:
            return pair
        else:
            return None

    def fit(self, corpus):

        if isinstance(corpus, str): # corpus can be list of strings or a string
            corpus = [corpus]

        self._initialize(corpus)

        # print('Initial vocab and words')
        # print(f'vocab = {self.vocab}')
        # print(f'words = {self.words}')

        merge_count = 0
        while(len(self.vocab) < self.vocab_size and merge_count < self.merge_count):
            pair = self._get_most_freq_adj_pair() # pair of tokens to merge in this step
            if pair:
                self._merge(pair)
                self.vocab.append(pair[0] + pair[1])
                merge_count += 1
            else:
                break
            if len(self.vocab) % 1000 == 0:
                self.save_vocab(f'vocab_{len(self.vocab)}')
            
            print(f'\r[info] Vocabulary size = {len(self.vocab)} | Merge count = {merge_count}', end='', flush=True)
        
        self.vocab = self.vocab + ['<UNK>', '<PAD>', '<CLS>', '<SEP>', '<MASK>'] # adding additonal tokens to vocab

        # print('Final vocab')
        # print(f'Vocab = {self.vocab}')
        # print(f"Final vocab size = {len(self.vocab)}")
        # print(f'Final merge count = {merge_count}')

    def _merge(self, pair):
        first, second = pair
        for k in range(len(self.words)):
            word = self.words[k]
            new_word = []

            i=0
            while(i < len(word) - 1):
                if word[i] == first and word[i+1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            if i == len(word) - 1: # add last token, if not merged
                new_word.append(word[i])

            self.words[k] = new_word

    def tokenize(self, text):
        words = self._preprocess(text)
        tokens = []
        for word in words:
            tokenized_word = self._apply_merges(word)
            tokens += tokenized_word
            # tokens += [word.replace('Ġ', '') for word in tokenized_word]

        return TokenizationOutput(self._post_process(tokens), self._encode(tokens))
    
    def _apply_merges(self, word):
        word = list(word) # ex: ĠHello --> ['Ġ','H','e','l','l','o']
        new_word = []
        i=0
        while(i < len(word)):

            token = word[i]

            if token in self.vocab:
                j = i + 1
                ext_token = token
                while(j < len(word)):
                    ext_token = ext_token + word[j]
                    if ext_token in self.vocab:
                        token = ext_token
                        i=j
                        j += 1  
                    else:
                        j += 1
                    
                new_word.append(token)

            else:
                # self.vocab.append(token) # adding new tokens to vocab (optional)
                new_word.append('<UNK>')
            i += 1

        return new_word

    def load_vocab(cls, file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            vocab = json.load(f)
            return vocab

    def save_vocab(self, file_path):
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(self.vocab, f)

    def _encode(self, tokens):
        return [self.vocab.index(token) for token in tokens]
    
    def decode(self, tokens):
        ids = tokens.ids
        text = ''
        for i, index in enumerate(ids):
            if i==0:
                text += self.vocab[index].replace('Ġ', '')
            else:
                text += self.vocab[index].replace('Ġ', ' ')
        return text