from tokenizer import BPETokenizer
import pandas as pd
from datetime import datetime


if __name__ == "__main__":

    # start = datetime.now()
    df = pd.read_csv("archive/bbc-news-data.csv", sep='\t')

    corpus = df['content'].tolist()



    # tokenizer = BPETokenizer(vocab_size = 100000, merge_count = 1000000)
    # tokenizer.fit(corpus)

    # tokenizer.save_vocab('vocab_final')

    # text = ' manchester united'
    # tokens = tokenizer.tokenize(text)
    # print()
    # print(tokens)

    # end = datetime.now()

    # total_time = (end - start).total_seconds()
    # print(f'Time taken = {total_time:.2f}s')

    t2 = BPETokenizer(vocab_file = 'vocab/vocab_final')
    # text = 'europe stake value ignored by America and India'
    # text = corpus[0]
    # print(text)
    

    max_len = 0
    sum=0
    for i, text in enumerate(corpus):
        output = t2.tokenize(text)
        
        if len(output.ids) > max_len:
            max_len = len(output.ids)
        sum += len(output.ids)
        avg = sum/(i+1)
        print(f'\r[info] index = {i} | max_len = {max_len} | avg = {avg:.0f}', end='', flush=True)

    # print(max_len)
    exit()
    print(output.tokens)
    print(output.ids)
    print(len(output.ids))

    decoded_text = t2.decode(output)
    print(decoded_text)