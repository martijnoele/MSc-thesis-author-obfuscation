import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import ExtraTreesClassifier
from pylab import *
from matplotlib2tikz import save as tikz_save
plt.style.use('ggplot')

blue = "#3D8DC1"
bronze = "#C28342"
green = "#52962B"

def visuals(df):
    return 1
    # author_sum(df)
    # plot_counts(df, 'words')
    # plot_counts(df, 'chars')
    # plot_counts(df, 'punctuations')
    # violin_plots(df)
    # plot_feature_ranking(df)
    # plot_features(df)


def author_sum(df):
    global blue, bronze, green
    author_summary = pd.DataFrame(df.groupby('author')['text'].count())
    author_summary.reset_index(inplace=True)
    plt.figure(figsize=(10,6))
    plt.title("Number of instances per author")
    sns.barplot(author_summary.author, author_summary.text, palette=[blue, bronze, green])
    plt.ylabel('Occurrences')
    plt.xlabel('Author')
    plt.draw()
    tikz_save('plots/authors.tikz',
              figureheight='\\figureheight',
              figurewidth='\\figurewidth')


def count_plot(df, attr):
    global blue, bronze, green
    sns.countplot(x=attr, hue="author", data=df, palette=[blue, bronze, green])

    plt.draw()


def plot_counts(df, col):
    global blue, bronze, green
    fig, axes = plt.subplots(nrows=3, sharex=True, sharey=True, figsize=(10, 7))
    fig.suptitle("Occurrences of " + col + " per author")
    sns.distplot(df.loc[df['author'] == 'EAP'][col].values, ax=axes[0], color=blue,
                 label='Edgar Allan Poe')
    sns.distplot(df.loc[df['author'] == 'MWS'][col].values, ax=axes[1], color=bronze,
                 label='Mary Shelley')
    sns.distplot(df.loc[df['author'] == 'HPL'][col].values, ax=axes[2], color=green,
                 label='HP Lovecraft')
    axes[0].legend(loc=0)
    axes[1].legend(loc=0)
    axes[2].legend(loc=0)
    plt.xlabel(col)
    plt.draw()

def violin_plots(df):
    global blue, bronze, green
    fig, axes = plt.subplots(3, 2, figsize=(10, 7), sharex=True, sharey=False)
    fig.suptitle("Visualization of features per author")
    sns.despine(left=True)
    sns.set_context("poster")
    sns.violinplot(x="author", y="punctuations", data=df, ax=axes[0, 0], palette=[blue, bronze, green])
    sns.violinplot(x="author", y="unique_words", data=df, ax=axes[0, 1], palette=[blue, bronze, green])
    sns.violinplot(x="author", y="stopwords", data=df, ax=axes[1, 0], palette=[blue, bronze, green])
    sns.violinplot(x="author", y="nouns", data=df, ax=axes[1, 1], palette=[blue, bronze, green])
    sns.violinplot(x="author", y="verbs", data=df, ax=axes[2, 0], palette=[blue, bronze, green])
    sns.violinplot(x="author", y="adjectives", data=df, ax=axes[2, 1], palette=[blue, bronze, green])
    sns.despine(left=True)
    plt.draw()


def plot_feature_ranking(df):
    global blue, bronze, green
    y = df['author'].astype(str)
    X = df.drop(['id', 'text', 'author'], axis=1)

    forest = ExtraTreesClassifier(n_estimators=250, random_state=123)
    forest.fit(X, y)
    importances = forest.feature_importances_
    std = np.std([tree.feature_importances_ for tree in forest.estimators_],
                 axis=0)
    indices = np.argsort(importances)[::-1]

    print("Feature ranking:")
    feature_names = list(X)
    sorted_feature_names = []
    for f in range(X.shape[1]):
        print("%d. feature %s (%f)" % (f + 1, feature_names[indices[f]], importances[indices[f]]))
        sorted_feature_names.append(feature_names[indices[f]])

    # Plot the feature importances
    plt.figure()
    plt.title("Importances of stylometric features")
    plt.bar(range(X.shape[1]), importances[indices],
            color=blue, yerr=std[indices], align="center")
    plt.xticks(range(X.shape[1]), sorted_feature_names, rotation='vertical')
    plt.draw()
    tikz_save('plots/feature_ranking.tikz',
              figureheight='\\figureheight',
              figurewidth='\\figurewidth')


def plot_features(df):
    df.pop('id')
    df.pop('text')
    fig, axes = plt.subplots(nrows=8, figsize=(10, 7), sharex=False, sharey=False)
    # fig.suptitle("Visualization of features per author")
    # sns.despine(left=True)
    i = 0
    for col in list(df):
        if col != 'author':
            axes[i].set_xlabel(col)
            sns.distplot(df.loc[df['author'] == 'EAP'][col].values, ax=axes[i], color=blue, label='Edgar Allan Poe')
            sns.distplot(df.loc[df['author'] == 'HPL'][col].values, ax=axes[i], color=bronze, label='HP Lovercraft')
            sns.distplot(df.loc[df['author'] == 'MWS'][col].values, ax=axes[i], color=green, label='Mary Shelley')
            i += 1

    handles, labels = axes[7].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right')
    plt.draw()
    tikz_save('plots/all_features.tikz',
              figureheight='\\figureheight',
              figurewidth='\\figurewidth')