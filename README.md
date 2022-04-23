# NLP - Text Classification for Online Orders
Classify online orders to one of 37 possible labels using shallow, hybrid, and deep approaches

Adjusted Mutual Information (AMI) score is the metric used to reflect how good the predicted classifications are.

## Shallow ML Classifiers
- Decision Tree
- Random Forest
- Logistic Regression
- CatBoost
- LGBM

Text preprocessing techniques applied:
- removing unwanted characters, numbers, stopwords
- tokenization
- lemmatization

Vectorizers used:
- CountVectorizer
- TF-IDF

Vectorizer hyperparameters tuned:
- min_df
- max_df
- ngram_range
- max_features

**Best competition AMI score using shallow ML:** 0.8389

## Hybrid
Since the LGBM algorithm yielded the best AMI scores among shallow ML classifiers, it was used for the hybrid approach.

- Sentence embeddings + LGBM
- Sentence embeddings + LGBM + text metadata

Hybrid approaches had better AMI scores from shallow ML algorithms when text preprocessing was not applied. Applying preprocessing prior to generating sentence embeddings yielded poor validation AMI scores (0.08-0.76).

Text metadata (i.e. number of words, nouns, and verbs in each document) were concatenated to the sentence embeddings but did not increase the AMI score.

**Best competition AMI score using hybrid approach with text metadata:** 0.9065

**Best competition AMI score using hybrid approach without text metadata:** 0.9077

## Deep
The deep approach focus was on transformers as they are considered to be the current state-of-the-art technique for NLP.
Preprocessing techniques did not help with deep approaches. Pre-trained language models worked best with unprocessed text.

The best performing pre-trained model was the RoBERTa-base model with an AMI score of 0.9381. BERT, DistilBERT, RoBERTa-large, and XLM-RoBERTa models were trained on the dataset but did not score higher than 0.934.

For the RoBERTa-base model, the learning rate, number of epochs, and training batch size were tuned using [W&B](https://wandb.ai/site/sweeps) library. An initial Bayes method sweep showed a learning rate trend where the optimal learning rate range is gleaned. Subsequent random method sweeps showed that the best validation AMI scores resulted when epochs were between 3 and 8 and batch size was between 25 and 50.

![Learning rate plot](README%20Figures/learning_rate_trend.png)

**Figure 1.** Bayesian method hyperparameter tuning results

#### What did the best model get wrong?
The best model often misclassified documents where two labels could apply. For example, \"*I would like to order 3 triple cheeseburgers and a diet coke*" can be labeled as *orderburgerintent* or *orderdrinkintent*.

![Incorrect predictions bar chart](README%20Figures/incorrect_predictions.png)
**Figure 2.** Incorrect validation set predictions

**Best competition AMI score using RoBERTa-base (learning rate: 4e-5, number of training epochs: 7, training batch size: 39):** 0.9381
