import retinasdk
from misc import bcolors, testfiles
import json
import pandas as pd
import numpy as np
from sys import stdout

liteClient = retinasdk.LiteClient("e29fcfe0")
fullClient = retinasdk.FullClient("your_api_key", apiServer="http://api.cortical.io/rest", retinaName="en_associative")


def compare_texts(texts1, texts2):
    print(bcolors.HEADER + "Compute similarity between sentences in dataframes:" + bcolors.ENDC)

    cosines = []
    i = 0
    l = len(texts1)

    for s1, s2 in zip(texts1, texts2):
        percent = i / l * 100
        stdout.write("\r{0:.3f} %".format(percent))
        stdout.flush()
        cosines.append(fullClient.compare(json.dumps([{"text": s1}, {"text": s2}])).cosineSimilarity)
        i += 1
    return cosines


def compare(filename):
    print(bcolors.OKBLUE + "Comparing original vs " + filename + bcolors.ENDC)
    df1 = pd.read_csv("data.nosync/test_original.csv")['text']
    df2 = pd.read_csv(filename)['text']

    print(df1.shape)
    print(df2.shape)
    cosines = compare_texts(df1, df2)

    print("\nResult report")
    print("Average cosine: {0:.3f}".format(np.mean(cosines)).replace(".", ","))
    print("Min. cosine: {0:.3f}".format(np.min(cosines)).replace(".", ","))
    print("Max. cosine: {0:.3f}".format(np.max(cosines)).replace(".", ","))
    print("Std. cosine: {0:.3f}".format(np.std(cosines)).replace(".", ","))


compare(testfiles.NOUNS_VERBS_MORE_PUNCT)
