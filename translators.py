import pandas as pd
from googletrans import Translator
from sklearn.model_selection import train_test_split


def load_original():
    print("Load original data")
    df = pd.read_csv("data.nosync/train.csv")
    y = df.pop('author')
    X = df.values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=123, stratify=y)
    df2 = pd.DataFrame(X_test, columns=['id', 'text'])
    df2['author'] = pd.Series(y_test).values

    return df2


def google_translate(df):
    print("Start translating one-way: EN-NL")
    for index, row in df.iterrows():
        print(index)
        translator = Translator()
        try:
            # translate the 'text' column
            translated = translator.translate(row['text'], src='en', dest='nl')
            row['text'] = translated.text
        except Exception as e:
            print(str(e))
            continue

    print("End of translation\n")
    print("Start translating two-way: NL-EN")

    for index, row in df.iterrows():
        print(index)
        translator = Translator()
        try:
            # translate the 'text' column
            translated = translator.translate(row['text'], src='nl', dest='en')
            row['text'] = translated.text
        except Exception as e:
            print(str(e))
            continue
    print("End of translation\n")
    return df


def translate_df():
    '''Function that opens original dataframe, translates all sentences (roundtrip) with Google Translate, and stores new dataframe to csv'''
    df = load_original()
    print(df.head())
    df_new = google_translate(df)
    print(df_new.head())

    df.to_csv("data.nosync/test_baseline_nl_short.csv")

translate_df()
