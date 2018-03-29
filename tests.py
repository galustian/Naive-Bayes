import pandas as pd
from GaussianNB import GaussianNB

if __name__ == '__main__':
    df = pd.read_csv('iris.csv', header=None)
    df = df.sample(frac=1)

    #df[4] = df[4].astype('category').cat.codes
    df[4] = df[4].map({'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2})
    
    X = df.iloc[:-35, :4].as_matrix()
    Y = df.iloc[:-35, -1].as_matrix()
    
    NB_classifier = GaussianNB()
    
    NB_classifier.train(X, Y)

    correct = 0
    for i in range(33, 0, -1):
        x_test = df.iloc[-i, :4].as_matrix()
        y_test = df.iloc[-i, -1]

        y_hat = NB_classifier.predict(x_test)
        
        if y_hat == y_test:
            correct += 1

        print("prediction:", y_hat, ", real value:", y_test)

    print("\nAccuracy:", correct / 33)
