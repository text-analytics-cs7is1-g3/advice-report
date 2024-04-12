import stanza
import pandas as pd
import numpy as np
import multiprocessing
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def is_you_subject(doc):
    print(".", end="", flush=True)
    for word in doc.sentences[0].words:
        if word.text.lower() == 'you' and (word.deprel == 'nsubj' or word.deprel == 'nsubj:pass') :
            return 1
    return 0

if __name__  == "__main__":
    # print(is_you_subject("You're the best"))
    # print(is_you_subject("It was you who killed the hag"))
    # print(is_you_subject("You were killed by the hag"))

    nlp = stanza.Pipeline(lang='en', processors="tokenize,mwt,pos,lemma,depparse", nthreads=multiprocessing.cpu_count() * 2)
    df = pd.read_csv("data/you_as_subject.csv")
    df["doc"] = nlp.bulk_process(df["document"])
    df["you_as_subject_stanza"] = df["doc"].apply(is_you_subject)


    print("accuracy", accuracy_score(df["you_as_subject"], df["you_as_subject_stanza"]))
    print("precision", precision_score(df["you_as_subject"], df["you_as_subject_stanza"]))
    print("recall", recall_score(df["you_as_subject"], df["you_as_subject_stanza"]))
    df.to_csv("data/you_as_subject_stanza.csv")
