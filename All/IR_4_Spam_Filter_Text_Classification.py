"""
Practical IR-4 : E-mail Spam Filtering using Text Classification (TF-IDF + Naive Bayes)
=======================================================================================
INSTRUCTIONS
 1. Install:  pip install numpy pandas matplotlib scikit-learn nltk
 2. Dataset : Enron_Spam_5000.csv (5000 real Enron e-mails, a sample of the
              Enron-Spam corpus: Category = ham/spam, Message = text). Option A
              reads it online from the datasets repo; Option B reads a local copy
              - see the two options in Step 2.
 3. Run     : python IR_4_Spam_Filter_Text_Classification.py
 4. Steps   : load -> clean (missing rows, duplicates) -> map ham/spam to 0/1
              -> stop words + Porter stemming -> 80/20 split
              -> TF-IDF (fit on train only) -> Multinomial Naive Bayes -> accuracy,
              precision, recall, F1, confusion matrix -> top spam words
              -> compare Logistic Regression and KNN
              -> predict 4 demo e-mails
"""

# --- Step 1: Import the libraries ---
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from nltk.stem import PorterStemmer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, classification_report

# --- Step 2: Load the dataset ---
# Option A: Fetch the dataset online from the datasets repo
url = "https://raw.githubusercontent.com/MrNobody4869/datasets/main/Enron_Spam_5000.csv"
df = pd.read_csv(url)

# Option B: Load from a local copy kept next to this script
# Download Enron_Spam_5000.csv from https://github.com/MrNobody4869/datasets
# Comment out Option A above and uncomment the line below to switch:
# df = pd.read_csv("Enron_Spam_5000.csv")
print("Data shape:", df.shape)

# --- Step 3: Clean the data and check class balance ---
# Drop missing rows and exact duplicate rows (this sample has none, so both are 0).
before = len(df)
df = df.dropna()
print("Missing rows removed:", before - len(df))
before = len(df)
df = df.drop_duplicates()
print("Exact duplicate rows removed:", before - len(df))
print("Rows left:", len(df))
# shows the class balance (this sample is nearly balanced)
print(df["Category"].value_counts())

# --- Step 4: Map the labels to numbers ---
# ham = 0, spam = 1 (spam is the "positive" class we want to catch).
df["label"] = df["Category"].map({"ham": 0, "spam": 1})

# --- Step 4b: Remove stop words and apply Porter stemming (as the manual asks) ---
# clean() works on one e-mail at a time and learns nothing from the data, so doing
# it before the split causes no data leakage.
# Stemming cuts word endings: "winning", "wins" -> "win".
stemmer = PorterStemmer()


def clean(t):
    kept = []
    for w in t.lower().split():
        if w not in ENGLISH_STOP_WORDS:
            kept.append(stemmer.stem(w))
    return " ".join(kept)


df["Message"] = df["Message"].apply(clean)

# --- Step 5: Train / test split (80 / 20, same spam ratio in both) ---
X_train, X_test, y_train, y_test = train_test_split(
    df["Message"], df["label"], test_size=0.2, random_state=42, stratify=df["label"])
print("Train:", len(X_train), "| Test:", len(X_test))

# --- Step 6: Turn text into TF-IDF numbers (fit on TRAIN only: no data leakage) ---
# stop words are already removed and words stemmed by clean()
tfidf = TfidfVectorizer()
# learn the vocabulary + IDF from train, then transform
X_train_vec = tfidf.fit_transform(X_train)
# test only transformed, never fitted
X_test_vec = tfidf.transform(X_test)
print("Vocabulary size:", len(tfidf.vocabulary_))

# --- Step 7: Train the main classifier - Multinomial Naive Bayes ---
nb = MultinomialNB(alpha=1.0)
nb.fit(X_train_vec, y_train)
y_pred = nb.predict(X_test_vec)

# --- Step 8: Evaluate ---
print("\nAccuracy :", round(accuracy_score(y_test, y_pred), 4))
print("Precision:", round(precision_score(y_test, y_pred), 4), "(spam)")
print("Recall   :", round(recall_score(y_test, y_pred), 4), "(spam)")
print("F1-score :", round(f1_score(y_test, y_pred), 4), "(spam)")
print(classification_report(y_test, y_pred, target_names=["ham", "spam"]))

# --- Step 9: Confusion matrix heatmap ---
cm = confusion_matrix(y_test, y_pred)
print("Confusion matrix [[TN FP] [FN TP]]:\n", cm)
tn, fp, fn, tp = cm.ravel()
# share of real ham that is correctly left alone
print("Specificity:", round(tn / (tn + fp), 4))
plt.imshow(cm, cmap="Blues")
plt.colorbar()
for i in range(2):
    for j in range(2):
        plt.text(j, i, cm[i, j], ha="center", va="center", fontsize=14)
plt.xticks([0, 1], ["ham", "spam"])
plt.yticks([0, 1], ["ham", "spam"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Naive Bayes - confusion matrix")
plt.show()

# --- Step 10: Top 10 words that point to spam and to ham ---
# log P(word | spam) - log P(word | ham): the bigger, the more spammy the word.
# Note: "escapenumber" is a placeholder the dataset authors put in place of every
# number, not a real word.
# It is common in both classes (a little more in ham), so it is NOT among the top
# spam words.
words = np.array(tfidf.get_feature_names_out())
score = nb.feature_log_prob_[1] - nb.feature_log_prob_[0]
print("\nTop 10 spam words:", list(words[np.argsort(score)[::-1][:10]]))
print("Top 10 ham words :", list(words[np.argsort(score)[:10]]))

# --- Step 11: Compare with Logistic Regression and KNN (same TF-IDF data) ---
# KNN is the algorithm named in the manual; cosine distance suits text.
models = [
    ("Logistic Regression", LogisticRegression(max_iter=1000)),
    ("KNN (k=5, cosine)", KNeighborsClassifier(n_neighbors=5, metric="cosine")),
]
for name, clf in models:
    p = clf.fit(X_train_vec, y_train).predict(X_test_vec)
    print(name, "-> accuracy", round(accuracy_score(y_test, p), 4),
          "| precision", round(precision_score(y_test, p), 4),
          "| recall", round(recall_score(y_test, p), 4))

# --- Step 12: Predict 4 new e-mails ---
# Demo input: hand-typed messages in the same plain lowercase style as the data
# (first 2 spam-like, last 2 normal).
demo = [
    "dear friend you have won a free prize claim your cash reward now "
    "click the link to buy cheap pills online",
    "limited offer save escapenumber percent on software and watches "
    "order now money back guarantee no prescription needed",
    "hi john please find attached the gas volumes for the meeting on monday "
    "let me know if the schedule works for you",
    "the project team will review the contract draft tomorrow morning "
    "please send your comments to the team before the call",
]
# demo e-mails get the same clean()
cleaned_demo = [clean(t) for t in demo]
probs = nb.predict_proba(tfidf.transform(cleaned_demo))[:, 1]
for text, pr in zip(demo, probs):
    print(("SPAM" if pr > 0.5 else "HAM "), round(pr, 3), "|", text)






























'''
================================================================================
VIVA AND THEORY NOTES - PRACTICAL IR-4 : E-MAIL SPAM FILTERING (TEXT CLASSIFICATION)
================================================================================

CODE EXPLANATION
----------------
1. Imports: numpy and pandas for data, matplotlib for the heatmap, sklearn for the split,
   TfidfVectorizer and ENGLISH_STOP_WORDS, three classifiers and the evaluation functions;
   nltk's PorterStemmer for stemming (it needs no download).
2. Load data: Option A (active) reads Enron_Spam_5000.csv from the datasets repo; Option B
   (commented) reads a local copy. Columns: Category (ham / spam) and Message (the e-mail text).
   The file is a random stratified sample of 5000 e-mails from the Enron-Spam corpus.
3. Clean: drop missing rows and drop exact duplicate rows (a repeated e-mail would otherwise
   appear in both train and test and make the score look better than it is). In this sample
   both counts are 0, which is fine: it only shows the data is already clean. Print the counts
   and the class balance (2547 ham, 2453 spam: almost balanced).
4. Labels: map ham -> 0 and spam -> 1 so the model works with numbers. Spam is the positive
   class (the one we want to catch).
5. Stop words + Porter stemming (Step 4b, as step 3 of the manual asks): clean(t) lower-cases
   an e-mail, drops every word in sklearn's ENGLISH_STOP_WORDS list and replaces each remaining
   word by its Porter stem ("winning", "wins" -> "win"). It is applied to the Message column
   BEFORE the split. That is safe (no data leakage) because it works on one e-mail at a time
   and learns nothing from the data, unlike TF-IDF. The Porter rules are explained in theory
   section 3 and in Viva Q1.
6. Split: 80 / 20, random_state=42, stratify=label so train and test have the same spam ratio.
7. TF-IDF: TfidfVectorizer() with no stop_words argument, because clean() already removed the
   stop words. fit_transform on TRAIN learns the vocabulary and the IDF weights; transform on
   TEST only applies them. This avoids data leakage. The vocabulary is now stems, and it is
   smaller (41792 words against 52314 without stemming).
8. Model: MultinomialNB(alpha=1.0) is trained on the train vectors and predicts the test set.
9. Metrics: accuracy, precision, recall and F1 for spam, plus classification_report.
10. Confusion matrix: drawn as a heatmap with plt.imshow; the numbers are written in the
    cells with plt.text. tn, fp, fn, tp = cm.ravel() unpacks the four cells and Specificity =
    tn / (tn + fp) is printed (the share of real ham left alone).
11. Top spam / ham words: feature_log_prob_ holds log P(word | class). Spam row minus ham row is
    big for words much more common in spam (viagra, pill, prescript, drug, medic ...) and very
    negative for words typical of ham (enron, ect, kaminski, mmbtu ...: real Enron business
    mail). Because of stemming these are now STEMS, not whole words: "pills" -> pill,
    "prescription" -> prescript, "Adobe" -> adob, "Vince" -> vinc, "Louise" -> louis. A stem
    does not have to be a real word. "escapenumber" is a placeholder the dataset authors wrote in place of every number;
    it is frequent in both classes (a little more in ham), so it is not a top spam word.
12. Comparison: Logistic Regression and KNN (k=5, cosine distance) trained on the same TF-IDF
    data; the manual names KNN, so it is shown for comparison.
13. Demo: 4 hand-typed e-mails (plain lowercase text, like the data) first go through the SAME
    clean() function (stop words + stemming), then the SAME fitted vectorizer, and are classified using predict_proba (probability of spam).

EXPECTED OUTPUT (real run, Option A data, seed 42)
----------------------------------------------------------------------
   Data shape: (5000, 2)
   Missing rows removed: 0
   Exact duplicate rows removed: 0        (nothing to remove in this sample)
   Rows left: 5000      (ham 2547, spam 2453)
   Train: 4000 | Test: 1000
   Vocabulary size: 41792      (52314 before stemming was added)
   Naive Bayes: Accuracy 0.966 | Precision 0.9914 | Recall 0.9389 | F1 0.9644 (spam)
   Confusion matrix [[TN FP] [FN TP]] = [[505 4] [30 461]]
   Specificity: 0.9921
   Top 10 spam words (stems): adob, viagra, pill, prescript, drug, medic, xp, ciali, med, php
   Top 10 ham words  (stems): enron, ect, vinc, hou, louis, kaminski, cc, pm, mmbtu, hpl
   Logistic Regression -> accuracy 0.965 | precision 0.9506 | recall 0.9796
   KNN (k=5, cosine)   -> accuracy 0.902 | precision 0.9735 | recall 0.8228
   Demo e-mails (P(spam)): 0.978 SPAM, 0.982 SPAM, 0.003 HAM, 0.015 HAM
   (A heatmap window opens for the confusion matrix. classification_report is also printed.)
   Only 4 real ham e-mails were marked spam (precision 0.9914); 30 spam got through.
   Stemming shrank the vocabulary by about 20 percent but did not raise the score here: without
   it the same model gave accuracy 0.974 and recall 0.9552 (22 spam missed). Run time about 15 s.
   On the full 28,063-email file the same pipeline WITHOUT stemming gave accuracy of about 0.983.

OUTPUT GRAPHS EXPLAINED
-----------------------
One figure only: the confusion-matrix heatmap (Figure 1, Step 9). Everything else is text.

GRAPH 1 : CONFUSION MATRIX HEATMAP (title "Naive Bayes - confusion matrix")
- Plots: a 2 x 2 table of how the 1000 test e-mails were classified, with a count in each cell.
- Rows (Actual) are the TRUE label, ham on top and spam below. Columns are what the model said.
- Colour (Blues bar): dark blue = many e-mails, almost white = very few.
- The diagonal (top-left, bottom-right) is correct; the other two cells are mistakes.
- Top-right = false positive (good mail sent to spam).
- Bottom-left = false negative (spam missed).
- OUR result: 505 ham kept, 4 ham blocked, 30 spam missed, 461 spam caught; accuracy 0.966.
- If asked: "Rows are the true class, columns the predicted class; few good mails are lost."
- Misreading: rows are TRUE labels, columns PREDICTED, do not swap them.

DETAILED THEORY NOTES
---------------------
1. WHAT IS SPAM FILTERING / TEXT CLASSIFICATION?
   - Definition: text classification assigns a text to one of several classes. Spam filtering is
     the two-class case: spam (unwanted bulk mail) or ham (normal mail).
   - Type: SUPERVISED learning. We train on messages that already have the correct label.
     (Answer to the manual's question: it is supervised, not unsupervised.)
   - Intuition: spam uses typical words (free, win, prize, viagra, lottery, order now). The model learns
     how likely each word is in spam and in ham and combines the evidence for a new message.
   - Maths in words (Naive Bayes): by Bayes theorem
        P(spam | words) is proportional to P(spam) * P(word1 | spam) * P(word2 | spam) * ...
     and the same for ham. The class with the larger value wins. "Naive" means we pretend the
     words are independent given the class, which is not true but works well. Multiplying many
     small numbers underflows, so sklearn adds logarithms instead.
   - Smoothing: alpha=1.0 (Laplace) adds 1 to every word count so a word never seen in spam
     does not make P(word | spam) = 0 and kill the whole product.
   - TF-IDF: TF = how often a word is in this message; IDF = log of (total messages / messages
     containing the word). TF * IDF is high for a word that is frequent here but rare overall,
     and low for common words such as "the" or "you". (scikit-learn actually computes
     idf = ln((1+N)/(1+df)) + 1 with smooth_idf=True, then L2-normalises each row.)

2. TYPES AND VARIANTS
   - Naive Bayes family : Multinomial (word counts / TF-IDF), Bernoulli (word present or not),
     Gaussian (continuous features).
   - Other classifiers for text : Logistic Regression, Linear SVM, KNN, Decision Tree / Random
     Forest, neural networks and transformers (BERT).
   - Filtering techniques (from the manual): content based, case based, heuristic / rule based,
     previous-likeness based (KNN), adaptive.
   - Features : Bag of Words (counts), TF-IDF, n-grams, word embeddings.

3. KEY CONCEPTS AND TERMS
   - Ham : legitimate message. Spam : unwanted message. Spam is our positive class.
   - Tokenization : splitting text into words. Stop words : very common words (the, is, and)
     removed by clean() (ENGLISH_STOP_WORDS).
   - Bag of Words : represent text as word counts, ignoring order.
   - TF-IDF vector : one number per vocabulary word for every message (sparse matrix).
   - Vocabulary : the set of distinct words learnt from the training data.
   - Prior P(class) : how common the class is. Likelihood P(word | class). Posterior
     P(class | message).
   - alpha (Laplace smoothing) : the pseudo-count added to each word.
   - Data leakage : test information sneaking into training. Fitting the vectorizer on the whole
     data would use test words and their IDF values during training, making results too good.
   - Imbalanced data : if one class is much bigger, plain accuracy can fool us. (Our Enron
     sample is almost balanced, but real inboxes and the SMS file are not.)
   - escapenumber : placeholder token that replaces every number in this dataset.
   - Porter stemmer (manual's question) : rule based algorithm that cuts word endings (-ed,
     -ing, -ness ...) to a common stem, so "winning" and "wins" become "win". It IS used in
     this code (Step 4b, clean()). Its rules are in Viva Q1. The stem need not be a word
     ("happy" -> happi, "prescription" -> prescript).
   - Stem : the form a word is cut down to by the stemmer (connected, connecting -> connect).

4. EVALUATION METRICS
   - Confusion matrix : TN (ham kept), FP (ham wrongly sent to spam), FN (spam let through),
     TP (spam caught).
   - Accuracy = (TP + TN) / all. Misleading on imbalanced data (a model that says "ham" for
     everything on a 87 % ham inbox would be 87 % accurate yet catch no spam). Here the classes
     are balanced (51 % ham), so accuracy is fair, but we still report precision and recall.
   - Precision = TP / (TP + FP). Of the messages marked spam, how many truly are spam. This is
     the most important number for a spam filter: a false positive means an IMPORTANT e-mail
     (job offer, bank, exam result) goes to the spam folder and may never be read. Users
     tolerate a few spam mails in the inbox more than a lost real mail.
   - Recall = TP / (TP + FN). Of all spam, how many we caught.
   - F1 = 2 * precision * recall / (precision + recall). One number balancing both.
   - Specificity = TN / (TN + FP) (true negative rate), listed in the manual. The code prints it
     with tn, fp, fn, tp = cm.ravel(): 505 / (505 + 4) = 0.9921, so 99.2 % of real ham is left
     alone.
   - classification_report prints precision, recall, F1 and support for each class.

5. DATA SPLIT / RELEVANT SETUP
   - 80 % train / 20 % test, random_state=42 so the split repeats, stratify=label keeps the same
     spam percentage in both parts.
   - The vectorizer is fitted on train only. Duplicates are removed BEFORE splitting so the same
     message cannot be in both train and test (none were found in this sample).
   - Naive Bayes needs no feature scaling.

6. THE DATASET
   - Enron-Spam (Metsis, Androutsopoulos and Paliouras, 2006). It is built from the real mail of
     6 Enron employees (folders enron1 ... enron6): their genuine ham is mixed with spam from
     spam traps and honeypots. It is a standard benchmark for spam filtering because the ham is
     real business e-mail (not made-up) and it is free to download. The manual shows enron1
     as its example: 3672 ham and 1500 spam.
   - The original corpus is hundreds of MB of raw mail files, one file per e-mail. A processed
     CSV of 28,063 e-mails (Category, Message) is on Kaggle (bayes2003/emails-for-spam-or-ham-
     classification-enron-2006). To keep the practical light we use Enron_Spam_5000.csv, a
     random STRATIFIED sample of 5000 rows from it: 2547 ham and 2453 spam. Results on the full
     file are about 0.983 accuracy; our sample gave 0.974 (before stemming was added), so the sample is representative.
   - Columns: Category (ham / spam) and Message. All rows are unique and none are empty.
   - The text is already pre-processed by the dataset authors: lowercase, punctuation removed,
     every number replaced by the token "escapenumber", and no headers or subject line.
   - These are real e-mails, long and formal (business talk about gas, meetings, contracts),
     not short SMS texts. Spam is about pills, software, lottery and stock tips.
   - Because the ham all comes from Enron staff, the model also learns Enron-only words
     (enron, ect, kaminski, mmbtu) as ham signals. Such a filter would work worse on another
     company's mail (dataset bias).
   - The manual also mentions the TREC corpora; the idea is the same: labelled mail in two
     classes. The manual's own algorithm for this assignment is KNN; we use Naive Bayes as the
     main model and show KNN for comparison.

7. ADVANTAGES
   - Naive Bayes is very fast to train and predict, simple, and works well on text.
   - Needs little training data, handles thousands of features (words).
   - Gives probabilities. Easy to update when new mail arrives.
   - TF-IDF is simple and needs no deep learning.

8. DISADVANTAGES
   - Independence assumption is not true (the words "free" and "prize" go together).
   - Ignores word order and meaning ("not good" vs "good").
   - Spammers change spelling (fr33, v1agra) to fool word lists.
   - Needs retraining as spam style changes (concept drift).
   - Zero-frequency problem without smoothing; sensitive to class imbalance priors.

9. REAL-WORLD USES
   - E-mail and SMS spam filters (Gmail, Yahoo, Outlook as in the manual's case study).
   - Sentiment analysis, news / topic classification, support-ticket routing.
   - Fake review and phishing detection, language identification.

10. COMPARISON WITH RELATED METHODS
   - Naive Bayes vs Logistic Regression : NB is a generative model, faster, good with little
     data; LR is discriminative and often a little more accurate with enough data.
   - Naive Bayes vs KNN : KNN has no training but is slow at prediction (compares with all
     training mails) and needs a good distance (cosine for text); NB is far faster.
   - Naive Bayes vs SVM : linear SVM is usually among the most accurate text classifiers; NB is
     simpler and gives probabilities.
   - Naive Bayes vs deep models (BERT) : deep models understand context and are more accurate
     but need much more data and computing power.
   - Bag of Words vs TF-IDF : counts treat every word equally; TF-IDF down-weights common words.

LIBRARIES AND FUNCTIONS USED
----------------------------
- numpy (np) : argsort gives the order of scores; array makes the word list indexable.
- pandas (pd) : read_csv loads the file; dropna removes missing rows;
  drop_duplicates removes exact repeats; value_counts shows class counts; map converts labels.
- matplotlib.pyplot (plt) : imshow (heatmap), colorbar, text (numbers in cells), xticks / yticks,
  xlabel / ylabel / title, show.
- sklearn.model_selection.train_test_split : test_size=0.2, random_state=42, stratify.
- nltk.stem.PorterStemmer : stemmer.stem(word) cuts a word to its Porter stem.
- sklearn.feature_extraction.text.ENGLISH_STOP_WORDS : sklearn's built-in stop-word list (a
  frozenset); clean() skips every word found in it.
- cm.ravel() : flattens the 2 x 2 confusion matrix to tn, fp, fn, tp.
- str methods : lower() lowers the text, split() cuts it into words, " ".join(...) glues the
  stems back into one string; df["Message"].apply(clean) runs clean() on every row.
- sklearn.feature_extraction.text.TfidfVectorizer : TfidfVectorizer() with defaults (the stop
  words are removed earlier by clean()); fit_transform (learn + convert), transform (convert only),
  vocabulary_ (word -> column), get_feature_names_out (list of words).
- sklearn.naive_bayes.MultinomialNB : alpha=1.0 smoothing; fit, predict, predict_proba,
  feature_log_prob_ (log P(word | class)).
- sklearn.linear_model.LogisticRegression : max_iter=1000 so it converges.
- sklearn.neighbors.KNeighborsClassifier : n_neighbors=5, metric="cosine".
- sklearn.metrics : accuracy_score, precision_score, recall_score, f1_score, confusion_matrix,
  classification_report.

DETAILED VIVA Q&A
-----------------
Q1. What is Porter Stemmer's Algorithm?
A1. A rule based algorithm that removes common suffixes (-ed, -ing, -ness, -ful ...) to reduce
    a word to its stem, e.g. "connected", "connecting" -> "connect". It reduces vocabulary size
    and groups related words, so the classifier sees fewer, stronger features.
    How it works (Martin Porter, 1980): the word is changed by a fixed list of rewrite rules
    applied in 5 steps, in order, and a rule fires only if its condition holds. The condition
    uses the "measure" m, roughly the number of vowel-consonant groups in the stem (m = 0 for
    "tree", 1 for "trouble", 2 for "troubles"), or "the stem contains a vowel".
      Step 1 : plurals and -ed / -ing. SSES -> SS (caresses -> caress), IES -> I (ponies ->
               poni), S -> nothing (cats -> cat); -ED and -ING are cut if the rest has a vowel
               (plastered -> plaster, motoring -> motor); a final Y becomes I if the stem has
               a vowel (happy -> happi).
      Step 2 : long suffixes become shorter ones when m > 0: -ATIONAL -> -ATE, -TIONAL ->
               -TION, -IZER -> -IZE ...
      Step 3 : -ICATE -> -IC, -ATIVE -> nothing, -ALIZE -> -AL, -FUL -> nothing,
               -NESS -> nothing ... (hopeful -> hope, goodness -> good).
      Step 4 : final suffixes such as -AL, -ANCE, -ER, -IC, -ABLE, -ATE, -MENT are removed
               when m > 1 (relational -> relat, conditional -> condit).
      Step 5 : a final -E is removed when m > 1, and a final double L is shortened.
    Because it only cuts letters by rule (no dictionary), the stem need not be a real word
    ("happy" -> happi) and it can make mistakes, but it is fast. In this practical it is
    PorterStemmer().stem(word) from nltk, applied inside clean() before the split.

Q2. Why did you use the k-NN algorithm for this problem?
A2. The manual uses k-NN: a new mail is compared with the labelled mails and gets the majority
    label of its k nearest ones, which is simple to explain. In our code the main model is
    Naive Bayes because it is faster and better suited to text; KNN is run as a comparison.

Q3. Is this supervised learning or unsupervised learning?
A3. Supervised, because the training messages have known labels (ham / spam) and the model
    learns to predict the label of new messages.

Q4. What are the different algorithms that can replace k-NN here?
A4. Naive Bayes, Logistic Regression, Support Vector Machine, Decision Tree / Random Forest,
    Gradient Boosting, neural networks and transformer models such as BERT.

Q5. What steps can be taken to improve accuracy further?
A5. Stemming or lemmatization, n-grams (word pairs), tuning alpha / k, using more data,
    handling class imbalance, adding features (message length, number of capitals, links,
    digits), using a linear SVM or an ensemble, and cross-validation to tune.

Q6. What is text classification?
A6. Assigning a piece of text to one of a set of predefined classes using a model trained on
    labelled examples. Spam / ham is the two-class case.

Q7. What is TF-IDF?
A7. Term Frequency x Inverse Document Frequency. TF counts how often a word is in the message;
    IDF = log(N / df) is large for rare words. The product highlights words that characterise
    a message and reduces the weight of words found everywhere. (scikit-learn uses
    idf = ln((1+N)/(1+df)) + 1 with smooth_idf=True, then L2-normalises each row.)

Q8. How does Naive Bayes classify a message?
A8. It computes, for each class, prior x product of word probabilities given that class (in
    log form, a sum) and chooses the class with the larger score. Words like "free" and
    "prize" have high probability under spam and push the result to spam.

Q9. Why is it called "naive"?
A9. It assumes every word is independent of the others given the class. In real text words are
    related, so the assumption is naive, but the classifier still works well.

Q10. What does alpha=1.0 do?
A10. It is Laplace smoothing: one extra count is added for every word and class. A word that
     never appeared in spam training mails still gets a small non-zero probability, so one
     unseen word cannot make the whole probability zero.

Q11. Why is precision important for a spam filter?
A11. A false positive sends a real, maybe important, mail to the spam folder where it can be
     lost. Precision = TP / (TP + FP) measures exactly how often "spam" decisions are right,
     so we want it very high even if a few spam mails slip through (lower recall).

Q12. What is data leakage and where could it occur here?
A12. Leakage is when information from the test data influences training. If we fitted the
     TF-IDF on all messages, the vocabulary and IDF weights would include test messages. We
     therefore fit on X_train only and just transform X_test.

Q13. Why drop duplicates?
A13. The same message appearing many times may land in both train and test; the model then
     "remembers" it and the test score is too optimistic. Dropping exact duplicates before the
     split keeps the test honest. (This Enron sample has 0 duplicates, but the step is a good
     habit and the older SMS data had hundreds.)

Q14. Why is accuracy not enough?
A14. On imbalanced data (say 87 % ham) a useless model that always says ham would be 87 %
     accurate. Our Enron sample is nearly balanced (2547 ham, 2453 spam), so accuracy is less
     misleading, but precision, recall and F1 for the spam class still show the real quality,
     especially how many real mails are lost (false positives).

Q15. What does stratify=y do in train_test_split?
A15. It keeps the same ham / spam ratio in the train and test parts, so the small spam class is
     fairly represented in both.

Q16. What are stop words and why remove them?
A16. Very common words such as "the", "is", "and", "to". They carry little meaning and appear
     in both classes, so removing them reduces noise and vocabulary size.

Q17. How do you find the most spam-like words?
A17. nb.feature_log_prob_ has log P(word | class). Subtracting the ham row from the spam row
     gives a score per word; the largest scores are the top spam words.

Q18. What is a confusion matrix? Read the four cells.
A18. A 2 x 2 table of actual vs predicted: TN = ham predicted ham, FP = ham predicted spam,
     FN = spam predicted ham, TP = spam predicted spam.

Q19. Define sensitivity and specificity (manual).
A19. Sensitivity (recall) = TP / (TP + FN), the share of spam caught. Specificity = TN /
     (TN + FP), the share of ham correctly left alone.

Q20. Why cosine distance for KNN on text?
A20. TF-IDF vectors are long and sparse; message length changes the vector size. Cosine
     compares the angle (direction) of two vectors, so a long and short message about the same
     topic still look similar. Here TfidfVectorizer L2-normalises every row by default
     (norm='l2'), so all vectors have length 1 and Euclidean and cosine rank neighbours the
     same way. Cosine matters when vectors are not normalised (raw counts, for example).

Q21. Logistic Regression vs Naive Bayes in this practical?
A21. Compare the printed numbers. NB gives accuracy 0.966 with precision 0.9914; LR gives
     0.965 with higher recall (0.9796) but lower precision (0.9506). LR catches more spam but
     sends more real mail to spam, while NB trains faster, is simpler to explain and has the
     better precision, which matters most for a spam filter. KNN is the weakest here (0.902).

Q22. What is Bag of Words?
A22. Representing a text as counts of its words, ignoring grammar and order. TF-IDF is a
     weighted version of it.

Q23. What is the difference between fit_transform and transform?
A23. fit_transform learns the vocabulary and IDF from the data and converts it; transform only
     converts new data using what was already learnt. Test data and new e-mails use transform.

Q24. What is concept drift in spam filtering?
A24. Spammers keep changing their words and tricks, so a model trained on old mail slowly gets
     worse. It must be retrained on fresh data regularly.

Q25. What are the five spam filtering techniques in the manual?
A25. Content based, case based, heuristic (rule) based, previous-likeness based (KNN) and
     adaptive filtering.

Q26. How are real filters like Gmail built?
A26. They combine thousands of rules, sender reputation, URL and domain checks, OCR on images,
     user reports ("report spam") and machine learning models, and update continuously.

Q27. Why can the demo message "dear friend you have won a free prize claim your cash reward" be spam?
A27. Words such as claim, cash, reward, prize, pills, cheap, free have high spam scores in
     training, so their product pushes P(spam) above 0.5 (our demo gives 0.978).

Q28. Why did you use a 5000-row sample and not the whole Enron corpus?
A28. The original corpus is hundreds of MB of raw mail files, too heavy for a lab practical.
     The processed CSV has 28,063 e-mails; a random stratified sample of 5000 keeps the same
     ham / spam mix, loads and trains in seconds, and gives about the same result (accuracy
     0.974 on the sample, before stemming was added, against about 0.983 on the full file).

Q29. What is "escapenumber" in the data?
A29. A placeholder token. The dataset authors replaced every number in every e-mail with the
     word escapenumber while pre-processing, so the model sees "a number was here" but not the
     value. It is not a real word, and it appears in both ham and spam, so it is not a useful
     spam signal.

Q30. How is spam filtering on e-mail different from on SMS?
A30. E-mails are long, formal and have headers, links and attachments; SMS texts are short,
     informal and full of abbreviations (u, ur, txt) and phone numbers. Long text gives the
     model many words to use, so e-mail filters can be very precise, and the Enron data is
     almost balanced while SMS data is mostly ham. Also the Enron mail is from the early
     2000s, so it is a benchmark and not a modern filter.

Q31. Why are words like enron, ect and kaminski the top ham words?
A31. All the ham in Enron-Spam comes from Enron staff, so those company names and mail
     terms are very common in ham and absent in spam. It shows dataset bias: the filter learns
     "looks like Enron mail" as well as "does not look like spam".

Q32. Why did stemming not improve the score here?
A32. Stemming merges word forms, which shrinks the vocabulary (52314 -> 41792) but also throws
     away detail: different words can be cut to the same stem. On this data recall dropped from
     0.9552 to 0.9389 and accuracy from 0.974 to 0.966, while precision stayed about 0.991.
     Stemming helps most on small data; here the sample is large enough that exact words
     already work well.

PRACTICAL AND CODING VIVA QUESTIONS
-----------------------------------
Q1. Write the line that maps ham/spam to 0/1.
A1. df["label"] = df["Category"].map({"ham": 0, "spam": 1}).

Q2. What happens if you write tfidf.fit_transform(X_test) for the test data?
A2. The vectorizer is re-fitted on the test data: a different vocabulary, so the columns no
    longer match the trained model (error or wrong results) and it is data leakage. Always use
    transform for test data.

Q3. Change alpha from 1.0 to 0.1 (or 5). What happens?
A3. Smaller alpha trusts the counts more (can overfit rare words, recall may rise, precision may
    fall); larger alpha smooths more (more cautious, may miss spam). Try alpha in a loop and
    compare precision / recall.

Q4. How do you keep only words that appear in at least 2 messages?
A4. TfidfVectorizer(min_df=2). The vocabulary gets smaller.

Q5. How do you add bigrams (word pairs) as features?
A5. TfidfVectorizer(ngram_range=(1, 2)). Vocabulary grows a lot.

Q6. How do you print the probability that a new mail is spam?
A6. nb.predict_proba(tfidf.transform(["your text"]))[:, 1] gives P(spam).

Q7. How would you make the filter stricter so fewer real mails go to spam?
A7. Raise the threshold: predict spam only if predict_proba(...)[:, 1] > 0.9 instead of 0.5.
    Precision goes up, recall goes down.

Q8. How do you make the KNN use 11 neighbours?
A8. KNeighborsClassifier(n_neighbors=11, metric="cosine").

Q9. How do you remove stop words in the code?
A9. In clean(): keep a word only if it is not in sklearn's ENGLISH_STOP_WORDS list (the same
    idea as passing stop_words="english" to TfidfVectorizer, or filtering against NLTK's
    stopwords.words("english")). Because of this the vectorizer is called with no arguments.

Q10. How do you get the number of mails per class after cleaning?
A10. df["Category"].value_counts() (printed in Step 3): ham 2547, spam 2453.

Q11. How do you use Stemming before TF-IDF?
A11. It is already done in Step 4b: stemmer = PorterStemmer(); clean(t) stems every word that is
    not a stop word; df["Message"] = df["Message"].apply(clean); then the vectorizer runs on the
    stemmed text. New e-mails must pass through clean() too, as the demo does. Because clean()
    works on one e-mail alone, running it before the split is not data leakage.

Q12. How do you save the confusion matrix picture?
A12. Call plt.savefig("confusion_matrix.png") before plt.show().

Q13. How would you use only a smaller part of the data for a quick test?
A13. Take a random sample before the split, for example df = df.sample(1000, random_state=42).

Q14. Write the lines that print the specificity.
A14. tn, fp, fn, tp = cm.ravel() and then print("Specificity:", round(tn / (tn + fp), 4)).
     Here 505 / 509 = 0.9921.

END OF VIVA NOTES
'''
