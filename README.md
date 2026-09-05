# Twitter-sentiment-Analysis
The objective of this project is to demonstrate an end-to-end Natural Language Processing (NLP) pipeline, from raw text cleaning to feature extraction and sentiment classification. This project can be extended to real-world applications such as brand monitoring, customer feedback analysis, and social media trend tracking.

This notebook is a Twitter (airline) sentiment analysis pipeline, and here's what it actually produces at each stage:

**1. Data loading & preview**
Loads a CSV of airline tweets with columns like `tweet_id`, `airline_sentiment`, `airline_sentiment_confidence`, `text`, etc. `df.head()` shows the first rows (neutral/positive/etc. sentiment labels alongside tweet text).

**2. Preprocessing**
- Extracts `text` (features) and `airline_sentiment` (target).
- Label-encodes sentiment into `0, 1, 2` (likely negative/neutral/positive).
- Cleans tweets: lowercases, strips `@mentions` and URLs, removes non-letters.
- Tokenizes, removes stopwords, and lemmatizes.

**3. Train/test split & vectorization**
- `xtrain.shape` → **(11712,)**, `xtest.shape` → **(2928,)**
- TF-IDF vectorizes the text into a sparse matrix, converted to a DataFrame.
- Drops rare words appearing in <0.1% of documents, cutting features from ~9,550 down to **1,125**.

**4. Model training (XGBoost)**
Trains an `XGBClassifier` (200 estimators, max_depth=20) on the TF-IDF features.

**5. Evaluation — final results:**
- **Predictions:** array of class labels like `[0, 0, 0, ..., 1, 1, 2]`
- **Classification report:**

| Class | Precision | Recall | F1-score | Support |
|---|---|---|---|---|
| 0 | 0.89 | 0.81 | 0.85 | 2010 |
| 1 | 0.51 | 0.59 | 0.55 | 530 |
| 2 | 0.61 | 0.74 | 0.67 | 388 |

**Overall accuracy: 0.77 (77%)** on 2928 test samples.

- **Confusion matrix:**
```
[[1637,  255,  118],
 [ 149,  315,   66],
 [  49,   50,  289]]
```

- Finally, a **heatmap visualization** of the confusion matrix is rendered (using seaborn, `cmap='rainbow'`), showing where the model confuses classes 0 (likely "negative"), 1 ("neutral"), and 2 ("positive").

**Bottom line:** The model does well on the majority class (0, probably negative tweets — 89% precision) but struggles more with the minority classes (1 and 2), giving an overall accuracy of ~77%.
