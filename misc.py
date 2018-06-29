from sklearn.metrics import accuracy_score, classification_report, log_loss
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.externals import joblib

class testfiles:
    TEST_ORIGINAL = "data.nosync/test_original.csv"
    BASELINE_FI = "data.nosync/test_baseline_fi_short.csv"
    BASELINE_NL = "data.nosync/test_baseline_nl_short.csv"
    NO_PUNCT = "data.nosync/no_punctuation.csv"
    LESS_PUNCT = "data.nosync/less_punctuation.csv"
    MORE_PUNCT = "data.nosync/more_punctuation.csv"
    NOUNS = "data.nosync/spacy_nouns_replaced.csv"
    NOUNS_NO_PUNCT = "data.nosync/spacy_nouns_replaced_no_punctuation.csv"
    NOUNS_LESS_PUNCT = "data.nosync/spacy_nouns_replaced_less_punctuation.csv"
    NOUNS_MORE_PUNCT = "data.nosync/spacy_nouns_replaced_more_punctuation.csv"
    VERBS = "data.nosync/spacy_verbs_replaced.csv"
    VERBS_NO_PUNCT = "data.nosync/spacy_verbs_replaced_no_punctuation.csv"
    VERBS_LESS_PUNCT = "data.nosync/spacy_verbs_replaced_less_punctuation.csv"
    VERBS_MORE_PUNCT = "data.nosync/spacy_verbs_replaced_more_punctuation.csv"
    NOUNS_VERBS = "data.nosync/spacy_nouns_verbs_replaced.csv"
    NOUNS_VERBS_NO_PUNCT = "data.nosync/spacy_nouns_verbs_replaced_no_punctuation.csv"
    NOUNS_VERBS_LESS_PUNCT = "data.nosync/spacy_nouns_verbs_replaced_less_punctuation.csv"
    NOUNS_VERBS_MORE_PUNCT = "data.nosync/spacy_nouns_verbs_replaced_more_punctuation.csv"

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def apply_grid_cv(name, model, grid, X, y, folds=5, score="f1_macro"):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=123, stratify=y)

    clf = GridSearchCV(model, param_grid=grid, cv=folds, scoring=score)
    clf.fit(X_train, y_train)

    print("\nCV Gridsearch with {}-fold CV: {}".format(folds, name))
    print("Best estimator in this grid: \n\t{}".format(clf.best_estimator_))

    print("\nApplying best model to test-data:")
    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    logloss = log_loss(y_test, y_prob)
    report = classification_report(y_test, y_pred, digits=5)

    print("\n{} results: \n\tAccuracy: {}\n\tLog loss: {}\n\tReport: \n{}".format(name, accuracy, logloss, report))

    return clf.best_estimator_


def apply_model(model, X, y):
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)

    accuracy = accuracy_score(y, y_pred)
    logloss = log_loss(y, y_prob)
    report = classification_report(y, y_pred, digits=5)

    print("\nApply best_estimator_ to test data: \n\tAccuracy: {}\n\tLog loss: {}\n\tReport: \n{}".format(accuracy, logloss, report))
    print(y_prob[:10])


def save_model(model, filename):
    print("\t* Save best model: '" + bcolors.OKBLUE + filename + '.sav' + bcolors.ENDC + "'")
    joblib.dump(model, 'models/' + filename + '.sav')


def load_model(filename):
    print("\t* Load model: '" + bcolors.OKBLUE + filename + '.sav' + bcolors.ENDC + "'")
    return joblib.load('models/' + filename + '.sav')
