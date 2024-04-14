import stanza
import pandas as pd
import numpy as np
import multiprocessing
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def batch_is_you_subject(i):
    nlp = stanza.Pipeline(lang='en', processors="tokenize,mwt,pos,lemma,depparse", nthreads=multiprocessing.cpu_count() * 2)
    df = pd.read_csv("data/you_as_subject.csv")
    # df["you_as_subject_stanza"] = .apply(is_you_subject)
    chunk = df.loc[i, "document"]
    def is_you_subject(string):
        print(string)
        doc = nlp(string)
        for word in doc.sentences[0].words:
            if word.text.lower() == 'you' and (word.deprel == 'nsubj' or word.deprel == 'nsubj:pass') :
                print("TRUE: ", string)
                return 1
        print("FALSE: ", string)
        return 0
    return chunk.apply(is_you_subject)



if __name__  == "__main__":
    # print(is_you_subject("You're the best"))
    # print(is_you_subject("It was you who killed the hag"))
    # print(is_you_subject("You were killed by the hag"))

    df = pd.read_csv("data/you_as_subject.csv")
    num_processes = 1 # 
    chunks = np.array_split(df.index, num_processes)
    pool = multiprocessing.Pool(processes=num_processes)
    results = pool.map(batch_is_you_subject, chunks)
    pool.close()
    pool.join()
    df["you_as_subject_stanza"] = pd.concat(results)


    print("accuracy", accuracy_score(df["you_as_subject"], df["you_as_subject_stanza"]))
    print("precision", precision_score(df["you_as_subject"], df["you_as_subject_stanza"]))
    print("recall", recall_score(df["you_as_subject"], df["you_as_subject_stanza"]))
df.to_csv("data/you_as_subject_stanza.csv")
