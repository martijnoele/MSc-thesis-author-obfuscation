# MSc Thesis: Author Obfuscation by Automatically Rewriting Texts
*Author identification is a practice of text mining that is widely used. A problem that is concerned with author identification is the privacy issue. Authors of anonymous texts should not be exposed with existing identification tools and therefore author obfuscation methods are to be developed. In this thesis, authors are obfuscated by rewriting the original texts in an automatic manner. Models that manipulate the original text - called revision models throughout this work - that are used in this thesis are built from an artificial intelligence point of view and they try to mimic human behaviour (i.e. as if a human rewrites texts without having background knowledge about which style variation is specific for which author).*

*To test the level of obfuscation, an adversary is trained on stylometric features and term frequencies that are extracted from the raw data. No form of preprocessing is applied, which means that the variation in the original text is preserved and can be captured into the classification model. Revision models use information that is obtained by determining the importance of features, what means that features with more predictive value will be manipulated first. Revision models are evaluated on two aspects: firstly, they are compared to round-trip baseline models and secondly, they are evaluated on the level of semantic preservation. The experiments lead to a revision model that obfuscates the author more effectively compared to the baseline models, but they also show that rewriting sentences that preserve semantics is tough.*

## How to use
**Train and apply classifier** <br />
Use main_classifier.py with __TRAIN = False or __TRAIN = True

**Plot visuals** <br />
Use main_analysis.py

**Rewrite texts** <br />
Use main_translate.py and uncomment the relevant translator functions.