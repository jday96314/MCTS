import numpy as np
from sklearn.model_selection._split import _BaseKFold

# Workaround for the fact https://github.com/scikit-learn/scikit-learn/pull/28519 is not yet merged.
class GroupKFoldShuffle(_BaseKFold):
    def __init__(self, n_splits=5, *, shuffle=False, random_state=None):
        super().__init__(n_splits=n_splits, shuffle=shuffle, random_state=random_state)

    def split(self, X, y, groups):
        # Validate data shapes.
        assert X.shape[0] == y.shape[0]
        assert X.shape[0] == groups.shape[0]

        # Find the unique groups in the dataset.
        unique_groups = np.unique(groups)

        # Shuffle the unique groups if shuffle is true.
        if self.shuffle:
            np.random.seed(self.random_state)
            unique_groups = np.random.permutation(unique_groups)

        # Split the shuffled groups into n_splits.
        split_groups = np.array_split(unique_groups, self.n_splits)

        # For each split, determine the train and test indices.
        for test_group_ids in split_groups:
            test_mask = np.isin(groups, test_group_ids)
            train_mask = ~test_mask

            train_idx = np.where(train_mask)[0]
            test_idx = np.where(test_mask)[0]

            yield train_idx, test_idx