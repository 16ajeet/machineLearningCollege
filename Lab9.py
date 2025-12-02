"""
1. Naive Bayes Classifier
Dataset: SMS Spam Collection Dataset (spam.csv)
Question: Use the SMS Spam Collection dataset to build a text classification model that predicts 
whether a message is spam or not using the Naive Bayes algorithm. 
Tasks:
1. Clean and preprocess the text (lowercasing, stopword removal, optional stemming).
2. Convert text to numerical features using TF-IDF.
3. Train a Multinomial Naive Bayes classifier.
4. Report accuracy, precision, recall, and confusion matrix.
5. Show two examples where the model misclassified messages and explain possible reasons.

2. Decision Tree Classifier
Dataset: IRIS dataset (iris.csv)
Question: Using the IRIS dataset, build a Decision Tree Classifier to classify flower species based on sepal and petal measurements.
Tasks:
1. Load and split the dataset into training and testing sets.
2. Train a Decision Tree using Gini and Entropy criteria separately.
3. Compare model performance for both criteria using accuracy and classification report """


import pandas as pd
import numpy as np
import re
from collections import Counter
from math import log

# -----------------------------------------------------------
# STEP 1: LOAD DATA
# -----------------------------------------------------------
df = pd.read_csv("spam.csv", encoding="latin-1")
df = df.iloc[:, :2]  # keep first 2 columns only
df.columns = ["label", "text"]  # label -> spam/ham, text -> message
df = df.dropna()  # remove any rows with NaN values

# -----------------------------------------------------------
# STEP 2: CLEANING + PREPROCESSING
# -----------------------------------------------------------

def clean_text(t):
    t = t.lower()                                   # lowercase
    t = re.sub(r"[^a-z0-9\s]", " ", t)              # keep only alphabets + numbers
    return t

stopwords = set([
    "the","a","an","is","are","to","in","and","or","for","of",
    "on","at","this","that","it","be","you","me","i","we"
])

def remove_stopwords(t):
    return " ".join([w for w in t.split() if w not in stopwords])

df["clean"] = df["text"].apply(clean_text).apply(remove_stopwords)

# -----------------------------------------------------------
# STEP 3: TRAIN TEST SPLIT 
# -----------------------------------------------------------
np.random.seed(42)
indices = np.random.permutation(len(df))
train_size = int(0.8 * len(df))

train = df.iloc[indices[:train_size]]   #gets row of indices from 0 to train_size
test  = df.iloc[indices[train_size:]]

# -----------------------------------------------------------
# STEP 4: BUILD VOCABULARY
# -----------------------------------------------------------

all_words = " ".join(train["clean"]).split()    #words ikkatha
freq = Counter(all_words)   #word count

# keep words that appear >= 3 times to avoid huge memory usage
vocab = sorted([w for w,c in freq.items() if c >= 3])   #rare word hata do
v_index = {w:i for i,w in enumerate(vocab)}

# -----------------------------------------------------------
# STEP 5: TF (Term Frequency) FUNCTION
# -----------------------------------------------------------

def tf_vector(text):
    vec = np.zeros(len(vocab), dtype=np.float32)
    words = text.split()
    counts = Counter(words)
    for w, c in counts.items():
        if w in v_index:
            vec[v_index[w]] = c
    return vec

train_tf = np.array([tf_vector(t) for t in train["clean"]])

# -----------------------------------------------------------
# STEP 6: IDF + TF-IDF
# -----------------------------------------------------------
N = len(train)
df_counts = np.zeros(len(vocab), dtype=np.float32)

for text in train["clean"]:
    unique_words = set(text.split())
    for w in unique_words:
        if w in v_index:
            df_counts[v_index[w]] += 1

idf = (np.log((N + 1) / (df_counts + 1)) + 1).astype(np.float32)

train_tfidf = train_tf * idf

# -----------------------------------------------------------
# STEP 7: CALCULATE NAIVE BAYES PARAMETERS
# -----------------------------------------------------------
y_train = np.array([1 if x == "spam" else 0 for x in train["label"]])

prior_spam = np.mean(y_train == 1)
prior_ham = 1 - prior_spam

spam_sum = train_tfidf[y_train == 1].sum(axis=0) + 1  # Laplace smoothing
ham_sum  = train_tfidf[y_train == 0].sum(axis=0) + 1

spam_total = spam_sum.sum()
ham_total  = ham_sum.sum()

p_w_spam = spam_sum / spam_total
p_w_ham  = ham_sum / ham_total

# -----------------------------------------------------------
# STEP 8: PREDICTION FUNCTION
# -----------------------------------------------------------

def predict(text):
    tf = tf_vector(text)
    tfidf = tf * idf
    
    log_spam = log(prior_spam) + np.sum(tfidf * np.log(p_w_spam))
    log_ham  = log(prior_ham)  + np.sum(tfidf * np.log(p_w_ham))
    
    return 1 if log_spam > log_ham else 0

# -----------------------------------------------------------
# STEP 9: EVALUATION
# -----------------------------------------------------------
y_test = np.array([1 if x == "spam" else 0 for x in test["label"]])
preds = np.array([predict(t) for t in test["clean"]])

accuracy = (preds == y_test).mean()
tp = sum((preds==1)&(y_test==1))
tn = sum((preds==0)&(y_test==0))
fp = sum((preds==1)&(y_test==0))
fn = sum((preds==0)&(y_test==1))

precision = tp/(tp+fp+1e-9)
recall = tp/(tp+fn+1e-9)

print("ACCURACY:", accuracy)
print("PRECISION:", precision)
print("RECALL:", recall)
print("CONFUSION MATRIX:")
print(tp, fp)
print(fn, tn)

# -----------------------------------------------------------
# STEP 10: SHOW MISCLASSIFIED EXAMPLES
# -----------------------------------------------------------

mis_idx = np.where(preds != y_test)[0][:2]
print("\nTwo misclassified messages:\n")
for i in mis_idx:
    print("TEXT:", test.iloc[i]["text"])
    print("ACTUAL:", test.iloc[i]["label"])
    print("PREDICTED:", "spam" if preds[i]==1 else "ham")
    print()


# -----------------------------------------------------------
# STEP 1: LOAD DATA
# -----------------------------------------------------------
df = pd.read_csv("Iris.csv")

X = df.iloc[:, 1:5].values    # features
y = df.iloc[:, 5].values      # labels

# -----------------------------------------------------------
# STEP 2: TRAIN TEST SPLIT
# -----------------------------------------------------------
np.random.seed(42)
perm = np.random.permutation(len(df))
train_size = int(0.8 * len(df))

train_idx = perm[:train_size]
test_idx = perm[train_size:]

X_train, y_train = X[train_idx], y[train_idx]
X_test, y_test = X[test_idx], y[test_idx]

# -----------------------------------------------------------
# STEP 3: GINI & ENTROPY FUNCTIONS
# -----------------------------------------------------------


def entropy(y):
    classes, counts = np.unique(y, return_counts=True)
    probs = counts / len(y)
    return -np.sum(probs * np.log2(probs+1e-9))

# -----------------------------------------------------------
# STEP 4: SPLIT DATA
# -----------------------------------------------------------

def split_data(X, y, feature_idx, threshold):
    left_mask = X[:, feature_idx] <= threshold
    right_mask = X[:, feature_idx] > threshold
    return X[left_mask], y[left_mask], X[right_mask], y[right_mask]

# -----------------------------------------------------------
# STEP 5: DECISION TREE NODE
# -----------------------------------------------------------

class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, label=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.label = label

# -----------------------------------------------------------
# STEP 6: BUILD TREE
# -----------------------------------------------------------

def build_tree(X, y, depth=0):

    # If only one class left → pure leaf
    if len(set(y)) == 1:
        return Node(label=y[0])
    
    best_feature, best_thresh, best_score = None, None, 999

    current_entropy =  entropy(y)

    for feature_idx in range(X.shape[1]):
        thresholds = np.unique(X[:, feature_idx])

        for t in thresholds:
            Xl, yl, Xr, yr = split_data(X, y, feature_idx, t)
            if len(yl)==0 or len(yr)==0:
                continue

            # weighted impurity
            left_imp  =  entropy(yl)
            right_imp =  entropy(yr)

            score = (len(yl)/len(y))*left_imp + (len(yr)/len(y))*right_imp

            if score < best_score:
                best_feature = feature_idx
                best_thresh = t
                best_score = score

    if best_feature is None:
        values, counts = np.unique(y, return_counts=True)
        return Node(label=values[np.argmax(counts)])
        

    Xl, yl, Xr, yr = split_data(X, y, best_feature, best_thresh)

    left_node = build_tree(Xl, yl, depth+1 )
    right_node = build_tree(Xr, yr, depth+1)

    return Node(best_feature, best_thresh, left_node, right_node)

# -----------------------------------------------------------
# STEP 7: PREDICTION FUNCTION
# -----------------------------------------------------------

def predict_row(node, row):
    if node.label is not None:
        return node.label
    if row[node.feature] <= node.threshold:
        return predict_row(node.left, row)
    else:
        return predict_row(node.right, row)

def predict_tree(tree, X):
    return np.array([predict_row(tree, row) for row in X])

# -----------------------------------------------------------
# STEP 8: TRAIN BOTH TREES
# -----------------------------------------------------------


tree_entropy = build_tree(X_train, y_train)

pred_entropy = predict_tree(tree_entropy, X_test)


acc_entropy = np.mean(pred_entropy == y_test)

print("Entropy Accuracy:", acc_entropy)

print("Entropy Classification Report:")

classes = np.unique(y_test)

for c in classes:
    tp = np.sum((pred_entropy == c) & (y_test == c))
    fp = np.sum((pred_entropy == c) & (y_test != c))
    fn = np.sum((pred_entropy != c) & (y_test == c))
    precision = tp / (tp + fp + 1e-9)
    recall = tp / (tp + fn + 1e-9)
    print(f"Class {c}: Precision={precision:.2f}, Recall={recall:.2f}")