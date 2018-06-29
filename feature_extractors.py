import nltk
from nltk.corpus import stopwords
import string
import pandas as pd
import numpy as np
from misc import *
from sklearn.decomposition import TruncatedSVD

stopwords_en = set(stopwords.words("english"))

def extract_features(path):
    print("\t* Extract stylometric features for dataframe in '" + bcolors.OKBLUE + path + bcolors.ENDC + "'")
    df = pd.read_csv(path)

    df['words'] = df.apply(lambda row: count_words(row), axis=1)
    df['chars'] = df.apply(lambda row: count_chars(row), axis=1)
    df['punctuations'] = df.apply(lambda row: fraction_punctuations(row), axis=1)
    df['unique_words'] = df.apply(lambda row: fraction_unique_words(row), axis=1)
    df['stopwords'] = df.apply(lambda row: fraction_stopwords(row), axis=1)
    df['nouns'] = df.apply(lambda row: fraction_nouns(row), axis=1)
    df['verbs'] = df.apply(lambda row: fraction_verbs(row), axis=1)
    df['adjectives'] = df.apply(lambda row: fraction_adjectives(row), axis=1)

    return df


def apply_vectorizer(df, vect, svd, print_top_n=False, n=50):
    print("\t* Transform fitted vectorizer to text column")
    fitted_vect = vect.transform(df['text'])
    vect_df = pd.DataFrame(fitted_vect.toarray(), columns=vect.get_feature_names())
    # svd_vect = svd.transform(vect_df)
    # best_features = [vect.get_feature_names()[i] for i in svd.components_[0].argsort()[::-1]]
    # vect_df = pd.DataFrame(svd_vect, columns=best_features[0:svd.n_components])

    df_new = pd.concat([df, vect_df], axis=1)

    if print_top_n:
        top_n = get_top_n_terms(vect, vect_df, n)
        print(top_n)

    return df_new


def get_top_n_terms(vectorizer, df, n):
    feature_array = np.array(vectorizer.get_feature_names())
    sorted_vect = np.argsort(df.values).flatten()[::-1]
    return feature_array[sorted_vect][:n]


def count_words(row):
    text = row['text']
    return len(text.split(' '))


def count_chars(row):
    return len(row['text'])


def fraction_unique_words(row):
    text = row['text']
    splitted_text = text.split(' ')
    splitted_text = [''.join(c for c in s if c not in string.punctuation) for s in splitted_text]
    splitted_text = [s for s in splitted_text if s]
    word_count = len(splitted_text)
    unique_count = len(list(set(splitted_text)))
    return unique_count/word_count


def fraction_stopwords(row):
    text = row['text'].lower()
    splitted_text = text.split(' ')
    splitted_text = [''.join(c for c in s if c not in string.punctuation) for s in splitted_text]
    splitted_text = [s for s in splitted_text if s]
    word_count = len(splitted_text)
    stopwords_count = len([w for w in splitted_text if w in stopwords_en])
    return stopwords_count/word_count


def fraction_punctuations(row):
    text = row['text']
    char_count = len(text)
    punctuation_count = len([c for c in text if c in string.punctuation])
    return punctuation_count/char_count


def fraction_nouns(row):
    splitted_text = row['text'].split(' ')
    splitted_text = [''.join(c for c in s if c not in string.punctuation) for s in splitted_text]
    splitted_text = [s for s in splitted_text if s]
    word_count = len(splitted_text)
    pos_list = nltk.pos_tag(splitted_text)
    noun_count = len([w for w in pos_list if w[1] in ('NN', 'NNP', 'NNPS', 'NNS')])
    return noun_count/word_count


def fraction_adjectives(row):
    splitted_text = row['text'].split(' ')
    splitted_text = [''.join(c for c in s if c not in string.punctuation) for s in splitted_text]
    splitted_text = [s for s in splitted_text if s]
    word_count = len(splitted_text)
    pos_list = nltk.pos_tag(splitted_text)
    adj_count = len([w for w in pos_list if w[1] in ('JJ', 'JJR', 'JJS')])
    return adj_count/word_count


def fraction_verbs(row):
    splitted_text = row['text'].split(' ')
    splitted_text = [''.join(c for c in s if c not in string.punctuation) for s in splitted_text]
    splitted_text = [s for s in splitted_text if s]
    word_count = len(splitted_text)
    pos_list = nltk.pos_tag(splitted_text)
    verbs_count = len([w for w in pos_list if w[1] in ('VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ')])
    return verbs_count/word_count
