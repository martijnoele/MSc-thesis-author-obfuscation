from feature_extractors import extract_features, apply_vectorizer
from misc import *
import pandas as pd
from matplotlib.pyplot import show
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.utils import shuffle
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import MinMaxScaler

__TRAIN = False
__TEST_PATH = testfiles.NOUNS_VERBS_MORE_PUNCT

original_df = pd.read_csv("data.nosync/train.csv")
y = original_df['author']

# initialize vectorizers
print(bcolors.HEADER + "STARTING PREPARATION PHASE:" + bcolors.ENDC)
print("\t* Initialize vectorizer")
# vect = TfifdVectorizer()
# vect = CountVectorizer(stop_words="english", min_df=3)
vect = TfidfVectorizer(stop_words=None, min_df=2)

print("\t* Fit vectorizer to original data")
vect_df = vect.fit_transform(original_df['text'])

svd = TruncatedSVD(n_components=300, n_iter=7, random_state=42)
# svd = svd.fit(vect_df)

if __TRAIN:
    print(bcolors.HEADER + "\nSTARTING TRAIN PHASE:" + bcolors.ENDC)

    df = extract_features("data.nosync/train.csv")

    # apply vectorization and delete unnecessary columns
    df_vect = apply_vectorizer(df, vect, svd, print_top_n=False, n=50)
    X = df_vect.drop(['id', 'text', 'author'], axis=1)

    df_vect.pop('id')
    df_vect.pop('text')
    df_vect.pop('author')

    min_max_scaler = MinMaxScaler()
    X = min_max_scaler.fit_transform(df_vect.values)

    # train model
    nb_grid = {'alpha': [0.01, 0.1, 1.0]}
    best_estimator = apply_grid_cv("Naive Bayes", MultinomialNB(), nb_grid, X, y)
    save_model(best_estimator, 'tuned/nb_model_6_stylo_cv_english_3')

    # dt_grid = {'max_depth': [3, 5, 10, 20, 25]}
    # best_estimator = apply_grid_cv("Decision Tree", DecisionTreeClassifier(), dt_grid, X, y)
    # save_model(best_estimator, 'best_dt_model_lsa_6000')
    #
    # knn_grid = {'n_neighbors': [5, 10]}
    # best_estimator = apply_grid_cv("KNN", KNeighborsClassifier(), knn_grid, X, y)
    # save_model(best_estimator, 'best_knn_model_lsa_6000')

else:
    print(bcolors.HEADER + "STARTING TEST PHASE:" + bcolors.ENDC)

    df = extract_features(__TEST_PATH)
    df = shuffle(df, random_state=123)
    y = df['author']

    # apply vectorization and delete unnecessary columns
    df_vect = apply_vectorizer(df, vect, svd, print_top_n=False, n=50)

    df_vect.pop('id')
    df_vect.pop('text')
    df_vect.pop('author')

    min_max_scaler = MinMaxScaler()
    X = min_max_scaler.fit_transform(df_vect.values)

    # load and apply model
    model = load_model('tuned/nb_model_8_stylo_tfidf_none_2')
    print("\t* Apply model to test data")
    apply_model(model, X, y)

show()