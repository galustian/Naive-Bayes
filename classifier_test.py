import pandas as pd
from naive_bayes_classifier import NaiveBayesClassifier

if __name__ == '__main__':
    df = pd.read_csv('spam.csv', encoding='ISO-8859-1')
    df = df.sample(frac=1.0)

    spam_data, legit_data = [], []
    
    for _, row in df.iterrows():
        if row['v1'] == 'spam':
            spam_data.append(row['v2'])
        else:
            legit_data.append(row['v2'])

    NB_classifier = NaiveBayesClassifier()

    spam_train = spam_data[:int(len(spam_data)*2 / 3)]
    spam_test = spam_data[int(len(spam_data)*2 / 3):]

    legit_train = legit_data[:int(len(legit_data)*2 / 3)]
    legit_test = legit_data[int(len(legit_data)*2 / 3):]
    
    NB_classifier.train(spam_train, legit_train)

    spam_accuracy = 0
    legit_accuracy = 0
    
    for text in spam_test:
        prediction = NB_classifier.predict(text)
        spam_accuracy += prediction
    spam_accuracy /= len(spam_test)

    for text in spam_test:
        prediction = NB_classifier.predict(text)
        legit_accuracy += 1 - prediction
    legit_accuracy /= len(legit_test)

    print("Spam Text prediction accuracy:", spam_accuracy)
    print("Legit Text prediction accuracy:", legit_accuracy)