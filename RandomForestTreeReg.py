import random
from collections import *
from typing import *

import numpy as np
import pandas as pd


class MyForestReg:

    def __init__(
            self,
            n_estimators: int = 10,
            max_features: float = 0.5,
            max_samples: float = 0.5,
            random_state: int = 42,
            max_depth: int = 5,
            min_samples_split: int = 2,
            max_leafs: int = 20,
            bins: int = None,
            oob_score: str = None
    ):
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.max_samples = max_samples
        self.random_state = random_state
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_leafs = max_leafs
        self.bins = bins
        self.leafs_cnt = 0
        self.trees: List[MyTreeReg] = []
        self.fi = defaultdict(float)
        self.oob_score = oob_score
        self.oob_score_ = 0

    def fit(self, X: pd.DataFrame, y: pd.Series):
        random.seed(self.random_state)

        self.trees = []
        self.leafs_cnt = 0
        self.fi = defaultdict(float)
        for col in X.columns:
            self.fi[col] = 0.0

        oob_preds = defaultdict(list)

        for i in range(self.n_estimators):
            cols_idx = random.sample(list(X.columns), round(len(X.columns) * self.max_features))
            rows_idx = random.sample(range(len(y)), round(len(y) * self.max_samples))
            oob_idx = [j for j in range(len(y)) if j not in rows_idx]

            X_train = X.iloc[rows_idx][cols_idx]
            y_train = y.iloc[rows_idx]

            tree = MyTreeReg(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                max_leafs=self.max_leafs,
                bins=self.bins,
                n=len(y)
            )
            tree.fit(X_train, y_train)
            self.trees.append(tree)
            self.leafs_cnt += tree.leafs_cnt

            if oob_idx:
                X_oob = X.iloc[oob_idx][cols_idx]
                preds = tree.predict(X_oob)
                for idx, pred in zip(oob_idx, preds):
                    oob_preds[idx].append(pred)

        if self.oob_score is not None and oob_preds:
            y_true, y_pred = [], []
            for idx, preds in oob_preds.items():
                if preds:
                    y_true.append(y.iloc[idx])
                    y_pred.append(np.mean(preds))
            y_true = np.array(y_true)
            y_pred = np.array(y_pred)
            self.oob_score_ = self.__count_score(pd.Series(y_true), pd.Series(y_pred))

        for tree in self.trees:
            for feature, value in tree.fi.items():
                self.fi[feature] += value

    def predict(self, X: pd.DataFrame):
        res = []
        for _, row in X.iterrows():
            pred = 0
            for tree in self.trees:
                pred += tree.predict(row.to_frame().T)[0]
            res.append(pred / self.n_estimators)
        return np.array(res)

    def __count_score(self, y_true: pd.Series, y_pred: pd.Series):
        if self.oob_score is not None:
            if self.oob_score == "mse":
                return self.__get_mse(y_true, y_pred)
            if self.oob_score == "mae":
                return self.__get_mae(y_true, y_pred)
            if self.oob_score == "rmse":
                return self.__get_rmse(y_true, y_pred)
            if self.oob_score == "mape":
                return self.__get_mape(y_true, y_pred)
            if self.oob_score == "r2":
                return self.__get_r2(y_true, y_pred)

    def __get_mae(self, y_true: pd.Series, y_pred: pd.Series):
        return np.sum(np.abs(y_true - y_pred)) / len(y_true)

    def __get_mse(self, y_true: pd.Series, y_pred: pd.Series):
        return np.sum((y_true - y_pred) ** 2) / len(y_true)

    def __get_rmse(self, y_true: pd.Series, y_pred: pd.Series):
        return (np.sum((y_true - y_pred) ** 2) / len(y_true)) ** 0.5

    def __get_r2(self, y_true: pd.Series, y_pred: pd.Series):
        return 1 - np.sum((y_true - y_pred) ** 2) / np.sum((y_true - y_true.mean()) ** 2)

    def __get_mape(self, y_true: pd.Series, y_pred: pd.Series):
        return np.sum(np.abs((y_true - y_pred) / y_true)) / len(y_true) * 100


class TreeNode:
    def __init__(
            self,
            feature: str = None,
            split_value: float = None,
            left_node: 'TreeNode' = None,
            right_node: 'TreeNode' = None,
            depth: int = 0,
            value: float = None
    ):
        self.feature = feature
        self.split_value = split_value
        self.left_node = left_node
        self.right_node = right_node
        self.depth = depth
        self.value = value

    def is_leave(self):
        return self.value is not None


class MyTreeReg:

    def __init__(
            self,
            max_depth: int = 5,
            min_samples_split: int = 2,
            max_leafs: int = 20,
            bins: int = None,
            n: int = 0
    ):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_leafs = max_leafs
        self.root = None
        self.bins = bins
        self.n = n
        self.histogram = {}
        self.fi = defaultdict(int)
        self.leafs_cnt = 0
        self.sum = 0

    def fit(self, X: pd.DataFrame, y: pd.Series):
        self._build_histogram(X)
        self.root = self._build_tree(0, X, y, self.max_leafs)

    def _build_histogram(self, X: pd.DataFrame):
        if self.bins is None:
            return
        for feature in X.columns:
            unique_values = np.sort(X[feature].unique())
            if len(unique_values) <= self.bins - 1:
                split_points = (unique_values[1:] + unique_values[:-1]) / 2
                self.histogram[feature] = np.array(split_points)
            else:
                self.histogram[feature] = np.histogram(X[feature].values, bins=self.bins)[1][1:-1]

    def _get_split_points(self, X: pd.DataFrame, feature: str):
        if self.bins is None:
            unique_values = np.sort(X[feature].unique())
            return np.array((unique_values[1:] + unique_values[:-1]) / 2)
        else:
            return self.histogram[feature]

    def _get_feature_importance(self, y, left_y, right_y):
        left_mse = self.get_mse(left_y)
        right_mse = self.get_mse(right_y)
        return len(y) / self.n * (
                self.get_mse(y) - len(left_y) * left_mse / len(y) - len(right_y) * right_mse / len(y))

    def predict(self, X: pd.DataFrame):
        res = []
        for _, row in X.iterrows():
            curr: TreeNode = self.root
            while not curr.is_leave():
                if row[curr.feature] <= curr.split_value:
                    curr = curr.left_node
                else:
                    curr = curr.right_node
            res.append(curr.value)
        return np.array(res)

    def _build_tree(self, depth: int, X: pd.DataFrame, y: pd.Series, leaves_available: int) -> TreeNode:
        if (depth != 0 and leaves_available <= 1) or depth == self.max_depth or len(
                y) < self.min_samples_split or y.nunique() == 1:
            return self.build_leaf(y)

        feature, split_point, _ = self.get_best_split(X, y)
        if feature is None:
            return self.build_leaf(y)

        left_idx = X[X[feature] <= split_point].index
        right_idx = X[X[feature] > split_point].index
        if len(left_idx) == 0 or len(right_idx) == 0:
            return self.build_leaf(y)

        left_share = max(1, leaves_available - 1)
        left_node = self._build_tree(depth + 1, X.loc[left_idx], y.loc[left_idx], left_share)

        left_used = self.count_leafs(left_node)
        right_share = max(1, leaves_available - left_used)
        right_node = self._build_tree(depth + 1, X.loc[right_idx], y.loc[right_idx], right_share)
        self.fi[feature] += self._get_feature_importance(y, y.loc[left_idx], y.loc[right_idx])

        return TreeNode(
            feature=feature,
            split_value=split_point,
            left_node=left_node,
            right_node=right_node,
            depth=depth
        )

    def count_leafs(self, node: TreeNode = None):
        if node is None:
            node = self.root
        if node.is_leave():
            return 1
        return self.count_leafs(node.left_node) + self.count_leafs(node.right_node)

    def build_leaf(self, y: pd.Series):
        self.leafs_cnt += 1
        val = float(np.mean(y))
        self.sum += val
        return TreeNode(
            value=val
        )

    def get_best_split(self, X: pd.DataFrame, y: pd.Series):
        best_res = (None, -1, -1)
        for feature in X.columns:
            sorted_idx = X[feature].sort_values().index
            n = len(y)

            sorted_X: pd.DataFrame = X.loc[sorted_idx]
            sorted_y = y.loc[sorted_idx]

            split_points = self._get_split_points(X, feature)
            for split_point in split_points:
                y1 = sorted_y[sorted_X[feature] <= split_point]
                y2 = sorted_y[sorted_X[feature] > split_point]

                s0 = np.var(y)
                s1 = np.var(y1)
                s2 = np.var(y2)

                s = s0 - len(y1) / n * s1 - len(y2) / n * s2
                if s > best_res[-1]:
                    best_res = (feature, split_point, s)
        return best_res

    def get_mse(self, y: pd.Series):
        n = len(y)
        if n <= 1:
            return 0.0
        y_arr = y.values.astype(float)
        mu = np.mean(y_arr)
        return np.sum((y_arr - mu) ** 2) / n

    def print_tree(self):
        self._recursive_print_tree(self.root, 1)
        print(self.leafs_cnt)
        print(self.sum)

    def _recursive_print_tree(self, root: TreeNode, depth, side: str = ""):
        if root is None:
            return
        if root.is_leave():
            print(f"{' ' * depth}{side} = {root.value}")
        else:
            print(f"{' ' * depth}{root.feature} | {root.split_value}")
            self._recursive_print_tree(root.left_node, depth + 1, "left")
            self._recursive_print_tree(root.right_node, depth + 1, "right")
