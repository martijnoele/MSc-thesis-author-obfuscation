import pandas as pd
import string
from misc import bcolors
import spacy
from gensim.models import KeyedVectors

model = KeyedVectors.load_word2vec_format('word_embeddings/GoogleNews-vectors-negative300.bin', binary=True)
nlp = spacy.load('en')


def load_original(filename):
    print("Load data")
    df = pd.read_csv("data.nosync/" + filename + ".csv")

    return df


def save_translation(df, model_name):
    print("Save translated data to '" + bcolors.OKBLUE + "data.nosync/" + model_name + ".csv" + bcolors.ENDC + "'")
    df.to_csv("data.nosync/" + model_name + ".csv", index=False)


def less_punctuation(row):
    text = row['text']
    text = ''.join([c for c in text if c not in [',']])
    return text


def no_punctuation(row):
    text = row['text']
    text = ''.join([c for c in text if c not in string.punctuation])
    return text


def more_punctuation(row):
    text = row['text']
    text = text.replace('.', '...')
    return text


def replace_pos(row):
    pos = ['VERB']
    # pos = ['NOUN', 'NN', 'NNP']
    text = row['text']
    print("Processing: {}".format(text))
    doc = nlp(str(text))
    new_text = []

    for token in doc:
        if (token.pos_ in pos or token.is_stop) and token.orth_ in model:
            similar_words, _ = zip(*model.wv.most_similar(positive=[token.orth_]))

            similar_words = [w for w in similar_words if '_' not in w and list(nlp(w))[0].lemma_ != token.lemma_]

            alt_word = similar_words[0] if len(similar_words) > 0 else token.orth_
            new_text.append(alt_word)
        else:
            new_text.append(token.orth_)

    return "".join([" "+i if not i.startswith("'") and i not in string.punctuation else i for i in new_text]).strip()


def spacy_translation(df):

    print(df.head())
    df['text_new'] = df.apply(lambda row: replace_pos(row), axis=1)
    df['text'] = df['text_new']
    df.pop('text_new')

    return df


df = load_original("spacy_nouns_verbs_replaced")
# df = spacy_translation(df)

# df['text'] = df.apply(lambda row: no_punctuation(row), axis=1)
# df['text'] = df.apply(lambda row: less_punctuation(row), axis=1)
df['text'] = df.apply(lambda row: more_punctuation(row), axis=1)

save_translation(df, 'spacy_nouns_verbs_replaced_more_punctuation')
