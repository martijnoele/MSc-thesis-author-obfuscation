from visuals import visuals
from matplotlib.pyplot import show
from feature_extractors import extract_features

df = extract_features("data.nosync/train.csv")
print(df.loc[df['words'] == 861]['id'].values)

visuals(df)
show()
