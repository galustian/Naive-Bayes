import nltk

class NaiveBayesClassifier():
    num_spam = 0
    num_legit = 0

    spam_word_occurences = {}
    legit_word_occurences = {}

    missing_word_prob = 0.001
    
    # spam_list & legit_list are python lists with the data
    # they are called like that, because that is what NB classif. are mostly used for
    # ------------
    # 'words' is analogous to the given data
    # 'spam' is analogous to the class (label)
    def train(self, spam_list, legit_list, missing_word_prob=0.001):
        self.num_spam = len(spam_list)
        self.num_legit = len(legit_list)
        self.missing_word_prob = missing_word_prob

        unique_mail_words = []

        for mail in spam_list:
            mail = self.format_data(mail)
            for word in mail:
                if word not in self.spam_word_occurences and word not in unique_mail_words:
                    self.spam_word_occurences[word] = 1
                elif word not in unique_mail_words:
                    self.spam_word_occurences[word] += 1
                
                unique_mail_words.append(word)
            
            unique_mail_words = []
        
        for mail in legit_list:
            mail = self.format_data(mail)
            for word in mail:
                if word not in self.legit_word_occurences and word not in unique_mail_words:
                    self.legit_word_occurences[word] = 1
                elif word not in unique_mail_words:
                    self.legit_word_occurences[word] += 1
                
                unique_mail_words.append(word)
            
            unique_mail_words = []
    
    def predict(self, data_string):
        words = self.format_data(data_string)
        
        # Bayes-Theorem
        # P(spam|W1, W2...) = P(W1, W2...|spam)*P(spam)  /  P(W1, W2...)

        return self.prob_words_given_spam(words) * self.prob_spam() / (self.prob_words(words) + 1e-170)
        
    def prob_words_given_spam(self, words):
        probability = 1
        # P(W1|spam) = P(W1 ∩ spam) / P(spam)
        for w in words:
            if w in self.spam_word_occurences:
                probability *= self.spam_word_occurences[w] / self.num_spam
            else:
                probability *= self.missing_word_prob / self.num_spam
        
        return probability
    
    def prob_words_given_legit(self, words):
        probability = 1
        # P(W1|spam) = P(W1 ∩ spam) / P(spam)
        for w in words:
            if w in self.legit_word_occurences:
                probability *= self.legit_word_occurences[w] / self.num_legit
            else:
                probability *= self.missing_word_prob / self.num_legit
        
        return probability

    def prob_words(self, words):
        # P(W1, W2) = P(W1, W2|spam)*P(spam) + P(W1, W2|legit)*P(legit)
        return self.prob_words_given_spam(words) * self.prob_spam() + self.prob_words_given_legit(words) * self.prob_legit()

    def prob_spam(self):
        return self.num_spam / (self.num_spam + self.num_legit)
    def prob_legit(self):
        return self.num_legit / (self.num_spam + self.num_legit)

    @staticmethod
    def format_data(data):
        words = nltk.word_tokenize(data)
        return [w.lower() for w in words if w != ',' and w != '.']


if __name__ == '__main__':
    from sys import argv
    import pandas as pd

    df = pd.read_csv('spam.csv', encoding='ISO-8859-1')
    spam_data, legit_data = [], []
    
    for _, row in df.iterrows():
        if row['v1'] == 'spam':
            spam_data.append(row['v2'])
        else:
            legit_data.append(row['v2'])

    spam_train = spam_data[:int(len(spam_data)*2 / 3)]
    spam_test = spam_data[int(len(spam_data)*2 / 3):]
    legit_train = legit_data[:int(len(legit_data)*2 / 3)]
    legit_test = legit_data[int(len(legit_data)*2 / 3):]
    
    NB_classifier = NaiveBayesClassifier()
    NB_classifier.train(spam_train, legit_train)
    
    text = ' '.join(argv[1:])
    print("Probability of being Spam:", NB_classifier.predict(text))