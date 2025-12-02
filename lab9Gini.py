import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# ENTROPY & GINI
# ---------------------------------------------------------

def entropy(y):
    classes, counts = np.unique(y, return_counts=True)
    probs = counts / len(y)
    return -np.sum(probs * np.log2(probs + 1e-12))

def gini_impurity(y):
    classes, counts = np.unique(y, return_counts=True)
    probs = counts / len(y)
    return 1 - np.sum(probs ** 2)

def information_gain(y, left_y, right_y, criterion="gini"):
    total = len(y)
    if criterion == "gini":
        parent = gini_impurity(y)
        left = gini_impurity(left_y)
        right = gini_impurity(right_y)
    else:
        parent = entropy(y)
        left = entropy(left_y)
        right = entropy(right_y)

    weighted = (len(left_y) / total) * left + (len(right_y) / total) * right
    return parent - weighted

# ---------------------------------------------------------
# TREE NODE
# ---------------------------------------------------------

class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

# ---------------------------------------------------------
# DECISION TREE CLASSIFIER
# ---------------------------------------------------------

class DecisionTree:
    def __init__(self, max_depth=5, min_samples_split=2, criterion="gini"):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.criterion = criterion
        self.root = None

    def best_split(self, X, y):
        best_feature, best_threshold = None, None
        best_gain = -1
        n_features = X.shape[1]

        for feature in range(n_features):
            thresholds = np.unique(X[:, feature])
            for t in thresholds:
                left_idx = X[:, feature] <= t
                right_idx = X[:, feature] > t

                if len(y[left_idx]) == 0 or len(y[right_idx]) == 0:
                    continue

                gain = information_gain(y, y[left_idx], y[right_idx], self.criterion)
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
                    best_threshold = t

        return best_feature, best_threshold

    def build(self, X, y, depth=0):
        if (depth >= self.max_depth or
            len(y) < self.min_samples_split or
            len(np.unique(y)) == 1):

            classes, counts = np.unique(y, return_counts=True)
            return Node(value=classes[np.argmax(counts)])

        feature, threshold = self.best_split(X, y)
        if feature is None:
            classes, counts = np.unique(y, return_counts=True)
            return Node(value=classes[np.argmax(counts)])

        left_idx = X[:, feature] <= threshold
        right_idx = X[:, feature] > threshold

        left_child = self.build(X[left_idx], y[left_idx], depth + 1)
        right_child = self.build(X[right_idx], y[right_idx], depth + 1)

        return Node(feature=feature, threshold=threshold, left=left_child, right=right_child)

    def fit(self, X, y):
        self.root = self.build(X, y)

    def predict_one(self, x, node):
        if node.value is not None:
            return node.value
        if x[node.feature] <= node.threshold:
            return self.predict_one(x, node.left)
        else:
            return self.predict_one(x, node.right)

    def predict(self, X):
        return np.array([self.predict_one(row, self.root) for row in X])

# ---------------------------------------------------------
# MATPLOTLIB TREE PLOTTER (SAVES FILE ALSO)
# ---------------------------------------------------------

def plot_tree_matplotlib(node, save_path=None):
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.axis('off')

    def depth(n):
        if n.value is not None:
            return 1
        return 1 + max(depth(n.left), depth(n.right))

    max_depth = depth(node)

    def draw(n, x, y, dx):
        if n.value is not None:
            text = f"Leaf: {n.value}"
        else:
            text = f"X{n.feature} ≤ {n.threshold}"

        ax.text(x, y, text, ha="center", va="center",
                bbox=dict(boxstyle="round,pad=0.4", fc="lightblue"))

        if n.value is None:
            x_left = x - dx
            y_child = y - 1
            ax.plot([x, x_left], [y - 0.1, y_child + 0.4], color="black")
            draw(n.left, x_left, y_child, dx / 2)

            x_right = x + dx
            ax.plot([x, x_right], [y - 0.1, y_child + 0.4], color="black")
            draw(n.right, x_right, y_child, dx / 2)

    draw(node, 0, 0, 2 ** (max_depth - 1))

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"\nSaved Tree Plot at: {save_path}")

    plt.show()

# ---------------------------------------------------------
# IRIS DATASET WORKFLOW
# ---------------------------------------------------------

iris = pd.read_csv("Iris.csv")
X = iris.iloc[:, 0:4].values
y = iris.iloc[:, 4].values

idx = np.arange(len(X))
np.random.shuffle(idx)
split = int(len(X) * 0.8)

X_train, X_test = X[idx[:split]], X[idx[split:]]
y_train, y_test = y[idx[:split]], y[idx[split:]]

# ---------------------- Train Gini Tree ------------------
gini_tree = DecisionTree(max_depth=4, criterion="gini")
gini_tree.fit(X_train, y_train)

# ---------------------- Predictions ----------------------
y_pred = gini_tree.predict(X_test)
print("\nPredictions:", y_pred)
print("Actual:     ", y_test)

accuracy = np.mean(y_pred == y_test)
print("\nAccuracy:", accuracy)

# ---------------------- Plot + Save -----------------------
plot_tree_matplotlib(gini_tree.root, save_path="decision_tree_gini.png")
