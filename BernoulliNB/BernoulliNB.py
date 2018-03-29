import numpy as np
import nltk

class BernoulliNB():
    total_counts = 0
    label_counts = {}

    feature_counts_per_label = {}
    
    text_data = False

    # a missing feature would have a probability of zero (not realistic in the real world)
    missing_feature_prob = 0.001

    # text_data=True => assumes that the feature (for one row) is just a string of words
    # text_data=False => takes multiple features, each 0 or 1 => Bernoulli (for one row)
    # Y always has to be a numpy array
    def train(self, X, Y, text_data=False, missing_feature_prob=0.001):
        self.missing_feature_prob = missing_feature_prob
        self.total_counts = len(X)

        # label_counts
        Y_unique_counts = np.bincount(Y)
        for i, y in enumerate(Y_unique_counts):
            self.label_counts[i] = y
            self.feature_counts_per_label[i] = {}

        if text_data == False:
            # set feature_counts_per_label
            for row_i in range(len(X)):
                for i, x in enumerate(X):
                    self.feature_counts_per_label[Y[row_i]][i] += x

        else:
            self.text_data = True
            # set feature_counts_per_label
            for row_i in range(len(X)):
                word_tokens = self.format_data(X[row_i])
                label = Y[row_i]

                for word in word_tokens:
                    if word not in self.feature_counts_per_label[label]:
                        self.feature_counts_per_label[label][word] = 1
                    else:
                        self.feature_counts_per_label[label][word] += 1

    @staticmethod
    def format_data(data):
        words = nltk.word_tokenize(data)
        words = [w.lower() for w in words if w != ',' and w != '.']
        
        unique_words = []
        for w in words:
            if w not in unique_words:
                unique_words.append(w)
        
        return unique_words


    # takes str or np.ndarray  
    def predict(self, features):
        if self.text_data:
            features = self.format_data(features)
        elif not self.text_data:
            pass
        else:
            raise ValueError("predict takes either str or np.ndarray as argument, got", type(features))
        
        best_label = None
        best_prob = 0
        
        for label in self.label_counts:
            # Bayes-Theorem
            # P(label|f1, f2...) = P(f1, f2...|label)*P(label)  /  P(f1, f2...)
            prob = self.prob_features_given_label(features, label) * self.prob_label(label) / (self.prob_features(features) + 1e-170)
            if prob > best_prob:
                best_prob = prob
                best_label = label

        return best_label, best_prob

    
    def prob_features_given_label(self, features, label):
        # P(f|label) = P(f ∩ label) / P(label)
        prob = 1

        if self.text_data:
            for feat in features:
                if feat in self.feature_counts_per_label[label]:
                    prob *= self.feature_counts_per_label[label][feat] / self.label_counts[label]
                else:
                    prob *= self.missing_feature_prob / self.label_counts[label]
        
            return prob
        

        for feat, feat_counts in enumerate(features):
            if feat in self.feature_counts_per_label[label]:
                prob *= (self.feature_counts_per_label[label][feat] / self.label_counts[label]) ** feat_counts
            else:
                prob *= self.missing_feature_prob / self.label_counts[label]
        
        return prob

    
    def prob_label(self, label):
        return self.label_counts[label] / self.total_counts

    
    def prob_features(self, features):
        # P(f1, f2) = P(f1, f2|l1) * P(l1) + P(f1, f2|l2) * P(l2) + ...
        prob = 0
        for label in self.label_counts:
            prob += self.prob_features_given_label(features, label) * self.prob_label(label)
        return prob


if __name__ == '__main__':
    import pandas as pd

    df = pd.read_csv('../spam.csv', encoding='ISO-8859-1')
    df['v1'] = df['v1'].astype('category').cat.codes
    data, labels = [], []
    
    for _, row in df.iterrows():
        data.append(row['v2'])
        labels.append(row['v1'])

    data = data[:int(len(data) / 3)]
    labels = labels[:int(len(labels) / 3)]
    labels = np.array(labels)
    
    NB_classifier = BernoulliNB()
    NB_classifier.train(data, labels, text_data=True)

    print("No Spam:")
    print(NB_classifier.predict(data[-8]))
    print(NB_classifier.predict(data[-42]))
    print(NB_classifier.predict(data[-56]))
    print(NB_classifier.predict(data[-3]))
    
    print("Spam:")
    print(NB_classifier.predict(data[2]))
    print(NB_classifier.predict(data[5]))
    print(NB_classifier.predict(data[8]))
    print(NB_classifier.predict(data[9]))