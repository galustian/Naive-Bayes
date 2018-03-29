import numpy as np
from scipy.stats import binom
import nltk

# Intro to Multinomial Naive Bayes:
# http://scikit-learn.org/stable/modules/naive_bayes.html#multinomial-naive-bayes
class MultinomialNB:
    total_counts = 0
    label_counts = {}

    feature_counts_per_label = {}
    total_features_per_label = {}

    text_data = False

    # Y has to be a numpy array
    def train(self, X, Y, text_data=False):
        self.text_data = text_data
        self.total_counts = len(X)

        Y_unique_counts = np.unique(Y)
        Y_bins = np.bincount(Y)
        for i, y in enumerate(Y_unique_counts):
            self.label_counts[i] = Y_bins[i]
            self.feature_counts_per_label[i] = {}
            self.total_features_per_label[i] = 0
        
        
        if text_data == False:
            for i in range(len(X)):
                for feat_i, n_feat in enumerate(X[i]):
                    self.total_features_per_label[Y[i]] += n_feat
                    if feat_i in self.feature_counts_per_label[Y[i]]:
                        self.feature_counts_per_label[Y[i]][feat_i] += n_feat
                    else:
                        self.feature_counts_per_label[Y[i]][feat_i] = n_feat
        else:
            for i in range(len(X)):
                words = self.format_data(X[i])
                self.total_features_per_label[Y[i]] += len(words)
                for w in words:
                    if w not in self.feature_counts_per_label[Y[i]]:
                        self.feature_counts_per_label[Y[i]][w] = 1
                    else:
                        self.feature_counts_per_label[Y[i]][w] += 1
                    

    # if text_data=False, features has to be a numpy array
    def predict(self, features, smoothing=1, binomial=False):
        best_label = 0
        best_prob = 0

        features = self.format_data(features)
        
        for label in self.label_counts:
            # P(label|features) = P(features|label) * P(label) / P(features)
            prob = self.prob_features_given_label(features, label, smoothing=smoothing, binomial=binomial) * self.prob_label(label) / (self.prob_features(features, smoothing, binomial) + 1e-160)
            if prob > best_prob:
                best_prob = prob
                best_label = label
            #print("Label", label, "Prob. feat given label", self.prob_features_given_label(features, label, smoothing=smoothing, binomial=binomial) * self.prob_label(label))
            #print("Prob label", label, ":", self.prob_label(label))
        
        #print("Labelcounts:", self.label_counts, "Totalcounts:", self.total_counts)
        return best_label
    
    def prob_features_given_label(self, features, label, smoothing=1, binomial=False):
        prob = 1
        unique_words = []

        for i, feat in enumerate(features):

            if binomial:
                
                if self.text_data:
                    # binomial probability of a word is of course counted once
                    if feat in unique_words: continue

                    if feat in self.feature_counts_per_label[label]:
                        bernoulli_prob = (self.feature_counts_per_label[label][feat] + smoothing) / (self.total_features_per_label[label])
                        prob *= binom.pmf(features.count(feat), len(features), bernoulli_prob)
                    else:
                        bernoulli_prob = smoothing / (self.total_features_per_label[label] + smoothing*self.label_counts[label])
                        prob *= binom.pmf(features.count(feat), len(features), bernoulli_prob)
                    
                    unique_words.append(feat)

                else:
                    if i in self.feature_counts_per_label[label]:
                        bernoulli_prob = (self.feature_counts_per_label[label][i] + smoothing) / (self.total_features_per_label[label] + smoothing*self.label_counts[label])
                        prob *= binom.pmf(feat, features.sum(), bernoulli_prob)
                    else:
                        bernoulli_prob = smoothing / (self.total_features_per_label[label] + smoothing*self.label_counts[label])
                        prob *= binom.pmf(feat, features.sum(), bernoulli_prob)
            
            else: # bernoulli probability

                if self.text_data:
                    if feat in self.feature_counts_per_label[label]:
                        prob *= (self.feature_counts_per_label[label][feat] + smoothing) / (self.total_features_per_label[label] + smoothing*self.label_counts[label])
                    else:
                        prob *= smoothing / (self.total_features_per_label[label] + smoothing*self.label_counts[label])
                else:
                    if i in self.feature_counts_per_label[label]:
                        prob *= (self.feature_counts_per_label[label][i] + smoothing) / (self.total_features_per_label[label] + smoothing*self.label_counts[label])
                    else:
                        prob *= smoothing / (self.total_features_per_label[label] + smoothing*self.label_counts[label])

        return prob

    
    def prob_label(self, label):
        return self.label_counts[label] / self.total_counts

    def prob_features(self, features, smoothing=1, binomial=False):
        prob = 0
        for label in self.label_counts:
            prob += self.prob_features_given_label(features, label, smoothing=smoothing, binomial=binomial) * self.prob_label(label)
        return prob

    @staticmethod
    def format_data(text):
        words = nltk.word_tokenize(text)
        return [w.lower() for w in words if w != ',' and w != '.']
    

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
    
    NB_classifier = MultinomialNB()
    NB_classifier.train(data, labels, text_data=True)

    print("No Spam:")
    print(NB_classifier.predict(data[-8], smoothing=1, binomial=True))
    print(NB_classifier.predict(data[-42], smoothing=1, binomial=True))
    print(NB_classifier.predict(data[-56], smoothing=1, binomial=True))
    print(NB_classifier.predict(data[-3], smoothing=1, binomial=True))
    
    print("Spam:")
    print(NB_classifier.predict(data[2], smoothing=1, binomial=True))
    print(NB_classifier.predict(data[5], smoothing=1, binomial=True))
    print(NB_classifier.predict(data[8], smoothing=1, binomial=True))
    print(NB_classifier.predict(data[9], smoothing=1, binomial=True))