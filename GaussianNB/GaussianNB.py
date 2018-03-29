import numpy as np

class GaussianNB():
    total_feature_mean = {}
    total_feature_variance = {}
    
    label_feature_mean = {}
    label_feature_variance = {}

    prob_labels = {}

    def train(self, X, Y):
        # Calculate mean / variance for all labels
        total_X_mean = np.mean(X, axis=0)
        total_X_var = np.var(X, axis=0)
        
        for feat_i in range(X.shape[1]):
            self.total_feature_mean[feat_i] = total_X_mean[feat_i]
            self.total_feature_variance[feat_i] = total_X_var[feat_i]
        
        # Calculate mean / variance for distinct labels
        Y_unique = np.unique(Y)

        for label in Y_unique:
            self.label_feature_mean[label] = {}
            self.label_feature_variance[label] = {}

            row_idx = np.where(Y==label)
            X_rows_by_label = X[row_idx, :][0]
            self.prob_labels[label] = len(row_idx) / len(Y)
            
            X_mean = np.mean(X_rows_by_label, axis=0)
            X_var = np.var(X_rows_by_label, axis=0)
            
            for feat_i in range(X.shape[1]):
                self.label_feature_mean[label][feat_i] = X_mean[feat_i]
                self.label_feature_variance[label][feat_i] = X_var[feat_i]


    def predict(self, features):
        best_label = 0
        best_prob = 0

        for label in self.label_feature_mean:
            # P(label|features) = P(features|label) * P(label) / P(features)
            prob = self.prob_features_given_label(features, label) * self.prob_labels[label] / self.prob_features(features)
            if prob > best_prob:
                best_prob = prob
                best_label = label
        
        return best_label
    
    def prob_features_given_label(self, features, label):
        prob = 1

        for feat_i, feat in enumerate(features):
            prob *= self.gaussian(feat, self.label_feature_mean[label][feat_i], self.label_feature_variance[label][feat_i])
        
        return prob
    
    
    def prob_features(self, features):
        prob = 0

        for label in self.label_feature_mean:
            prob += self.prob_features_given_label(features, label) * self.prob_labels[label]

        return prob
    

    @staticmethod
    def gaussian(x, mean, var):
        return np.exp(-np.power(x-mean, 2) / (2*var)) / np.sqrt(2*np.pi*var)