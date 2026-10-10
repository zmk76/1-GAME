import gensim
import time
from gensim.models import Word2Vec
from tqdm import tqdm
import numpy as np

def train_line(user_item_dict, num_users, num_items,
                        nrl_pretrain_epochs=40, embedding_size=512):
 
    start_time = time.time()

    vocab_corpus = []

    for user in range(num_users):
        vocab_corpus.append(["u_{}".format(user)])
    
    for item in range(num_items):
        vocab_corpus.append(["i_{}".format(item)])

        
    corpus = []
    for user, items in user_item_dict.items():
        for item in items:
            user_id = "u_{}".format(user)
            item_id = "i_{}".format(item)

            corpus.append([user_id, item_id])
            corpus.append([item_id, user_id])

        
     
    print("start training word2vec")
    word2vec_model = Word2Vec(sentences=vocab_corpus, vector_size=embedding_size, window=2, min_count=0, workers=4)
    for i in tqdm(range(nrl_pretrain_epochs)):
        print("train word2vec epoch {}".format(i))
        word2vec_model.train(corpus, total_examples=len(corpus), epochs=1)


    user_embeddings = np.array([word2vec_model.wv["u_{}".format(i)] for i in range(num_users)])
    item_embeddings = np.array([word2vec_model.wv["i_{}".format(i)] for i in range(num_items)])


    print("nrl time: ", time.time() - start_time)


    return user_embeddings, item_embeddings