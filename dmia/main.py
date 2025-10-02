import matplotlib.pyplot as plt
import pandas as pd
import tqdm
from sklearn import linear_model
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from dmia.classifiers.logistic_regression import LogisticRegression
from dmia.gradient_check import *


def read_data(path):
    return pd.read_csv(path)

def get_predictions(train_df, review_summaries):
    #########################################################################
    #                      Извлечение признаков                             #
    #########################################################################
    vectorizer = TfidfVectorizer()
    tfidfed = vectorizer.fit_transform(review_summaries)

    X = tfidfed
    y = train_df.Prediction.values
    return train_test_split(X, y, train_size=0.7, random_state=42)

def gradient_check(X_train, X_test, y_train, y_test):
    X_train_sample = X_train[:10000]
    y_train_sample = y_train[:10000]
    clf = LogisticRegression()
    clf.w = np.random.randn(X_train_sample.shape[1] + 1) * 2
    loss, grad = clf.loss(LogisticRegression.append_biases(X_train_sample), y_train_sample, 0.0)

    f = lambda w: clf.loss(LogisticRegression.append_biases(X_train_sample), y_train_sample, 0.0)[0]
    grad_numerical = grad_check_sparse(f, clf.w, grad, 10)

    print(grad_numerical)

    clf = LogisticRegression()
    clf.train(X_train, y_train)

    print("Train f1-score = %.3f" % accuracy_score(y_train, clf.predict(X_train)))
    print("Test f1-score = %.3f" % accuracy_score(y_test, clf.predict(X_test)))

def draw_plot(X_train, X_test, y_train, y_test):
    ## Кривая обучения
    clf = LogisticRegression()
    train_scores = []
    test_scores = []
    num_iters = 1000

    for i in tqdm.trange(num_iters):
        clf.train(X_train, y_train, learning_rate=1.0, num_iters=1, batch_size=256, reg=1e-3)
        train_scores.append(accuracy_score(y_train, clf.predict(X_train)))
        test_scores.append(accuracy_score(y_test, clf.predict(X_test)))

    plt.figure(figsize=(10, 8))
    plt.plot(train_scores, 'r', test_scores, 'b')
    plt.show()



if __name__ == "__main__":
    train_df = read_data('../data/train.csv')

    review_summaries = train_df['Reviews_Summary'].to_list()
    review_summaries = [l.lower() for l in review_summaries]

    X_train, X_test, y_train, y_test = get_predictions(train_df, review_summaries)

    gradient_check(X_train, X_test, y_train, y_test)

    draw_plot(X_train, X_test, y_train, y_test)

    ## 4. HW
    vectorizer = TfidfVectorizer(max_features=3000)
    X = vectorizer.fit_transform(review_summaries)
    y = train_df.Prediction.values

    clf = LogisticRegression()
    clf.train(X, y, verbose=True, learning_rate=1.0, num_iters=1000, batch_size=256, reg=1e-3)

    # Получите индексы фичей
    pos_features = np.argsort(clf.w)[-5:]
    neg_features = np.argsort(clf.w)[:5]

    fnames = vectorizer.get_feature_names_out()

    print(f'\n##### Custom LogisticRegression #####\n')
    print([fnames[p] for p in pos_features])
    print([fnames[n] for n in neg_features])

    ### 5 Сравнение

    clf = LogisticRegression()
    clf.train(X_train, y_train, verbose=True, learning_rate=1.0, num_iters=1000, batch_size=256, reg=1e-3)

    print(f'\n##### Custom LogisticRegression #####\n')
    print("Train f1-score = %.3f" % accuracy_score(y_train, clf.predict(X_train)))
    print("Test f1-score = %.3f" % accuracy_score(y_test, clf.predict(X_test)))


    clf_sk = linear_model.SGDClassifier(
        max_iter=1000,
        random_state=42,
        loss="log_loss",
        penalty="l2",
        alpha=1e-3,
        eta0=1.0,
        learning_rate="constant"
    )
    clf_sk.fit(X_train, y_train)

    print(f'\n##### Sklearn LogisticRegression #####\n')
    print("Train accuracy = %.3f" % accuracy_score(y_train, clf_sk.predict(X_train)))
    print("Test accuracy = %.3f" % accuracy_score(y_test, clf_sk.predict(X_test)))
