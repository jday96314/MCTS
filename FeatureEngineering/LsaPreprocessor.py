from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import Pipeline
from lightgbm import LGBMRegressor

class LsaPreprocessor:
    def __init__(self, n_components=100, analyzer='word', ngram_range=(2, 3), max_df=1.0, min_df=0.0, kept_feature_count=10):
        self.n_components = n_components
        self.ngram_range = ngram_range
        self.analyzer = analyzer
        self.ngram_range = ngram_range
        self.max_df = max_df
        self.min_df = min_df
        self.kept_feature_count = kept_feature_count

    def fit_transform(self, lud_rules, train_df, labels):
        # COMPUTE LSA FEATURES.
        self.pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(analyzer=self.analyzer, ngram_range=self.ngram_range, max_df=self.max_df, min_df=self.min_df)),
            ('lsa', TruncatedSVD(n_components=self.n_components))
        ])
        self.pipeline.fit(lud_rules.unique())
        all_lsa_features = self.pipeline.transform(lud_rules)

        # SELECT TOP FEATURES.
        temp_train_df = train_df.copy()
        for feature_index in range(self.n_components):
            temp_train_df[f'lsa_{feature_index}'] = all_lsa_features[:, feature_index]

        regressor = LGBMRegressor(verbose=-1)
        regressor.fit(temp_train_df, labels)

        feature_importances = regressor.feature_importances_
        lsa_feature_ids_to_importances = [
            (importance_index, importance)
            for importance_index, importance in enumerate(feature_importances)
            if importance_index >= len(train_df.columns)
        ]

        lsa_feature_ids_to_importances.sort(key=lambda x: x[1], reverse=True)
        top_lsa_feature_ids = [feature_id for feature_id, _ in lsa_feature_ids_to_importances[:self.kept_feature_count]]
        self.top_lsa_feature_names = [temp_train_df.columns[feature_id] for feature_id in top_lsa_feature_ids]
        self.top_lsa_feature_indices = [int(feature_name.split('_')[-1]) for feature_name in self.top_lsa_feature_names]

        # ADD TOP FEATURES TO TRAIN DF.
        for feature_index, feature_name in zip(self.top_lsa_feature_indices, self.top_lsa_feature_names):
            train_df[feature_name] = all_lsa_features[:, feature_index]
        
        return train_df
    
    def transform(self, lud_rules, test_df):
        # COMPUTE LSA FEATURES.
        all_lsa_features = self.pipeline.transform(lud_rules)

        # ADD TOP FEATURES TO TEST DF.
        for feature_index, feature_name in zip(self.top_lsa_feature_indices, self.top_lsa_feature_names):
            test_df[feature_name] = all_lsa_features[:, feature_index]
        
        return test_df