import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

from sklearn.linear_model import LogisticRegression

from sklearn.utils import shuffle

import nltk
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('punkt')  # Only needs to be done once
nltk.download('punkt_tab')

from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.stem import LancasterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

traindata = pd.read_table("trainingset.txt")
testdata = pd.read_table("testset.txt")

print("Training Data Labels")
print(traindata['label'].value_counts())

print("Test Data Labels")
print(testdata['label'].value_counts())

# Extracting the relevant information from the datasets
traindata_X = traindata.iloc[:, [1, 4]]  # Taking tweetText and username
traindata_Y = traindata.iloc[:, 6]  # Taking the labels

testdata_X = testdata.iloc[:, [1, 4]]
testdata_Y = testdata.iloc[:, 6]

# Iteration 0.5 - No preprocessing, using all features
bow = CountVectorizer()
train_X = bow.fit_transform(traindata_X['tweetText'])

log = LogisticRegression()
log.fit(train_X, traindata_Y)

# test_X = bow.fit_transform(testdata_X['tweetText'])
# This iteration was a control iteration. It was expected to not work.


# Iteration 1 - Using Hashvectizer with 1000 features
featureExtract = HashingVectorizer(n_features=1000)

train_X = featureExtract.transform(traindata_X['tweetText'])
test_X = featureExtract.transform(testdata_X['tweetText'])

log.fit(train_X, traindata_Y)
print("Iteration 1 F1 score: ", log.score(test_X, testdata_Y))


# Iteration 2 - Using max_features in count vectorizer
bow = CountVectorizer(max_features=1000)
train_X = bow.fit_transform(traindata_X['tweetText'])
test_X = bow.fit_transform(testdata_X['tweetText'])
log.fit(train_X, traindata_Y)
print("Iteration 2 F1 score: ", log.score(test_X, testdata_Y))


# Iteration 3 - Using tfidf with 1000 features + normalising the features
tfidf = TfidfVectorizer(max_features=1000)  # Make smooth_idf false next
train_X = tfidf.fit_transform(traindata_X['tweetText'])
test_X = tfidf.fit_transform(testdata_X['tweetText'])
log.fit(train_X, traindata_Y)
print("Iteration 3 F1 score: ", log.score(test_X, testdata_Y))  # 0.39866844207723034

# Iteration 3.1 - Using tfidftransformer on the tfidfvectorized data
tfidf_transform = TfidfTransformer()
tfidf_transform.fit(train_X)
train_X = tfidf_transform.transform(train_X)
test_X = tfidf_transform.transform(test_X)
log.fit(train_X, traindata_Y)
print("Iteration 3.1 F1 score: ", log.score(test_X, testdata_Y))

# Iteration 3.2 - Using tfidf with 1000 features and false smooth idf
tfidf = TfidfVectorizer(max_features=1000, smooth_idf=False)
train_X = tfidf.fit_transform(traindata_X['tweetText'])
test_X = tfidf.fit_transform(testdata_X['tweetText'])
log.fit(train_X, traindata_Y)
print("Iteration 3.2 F1 score: ", log.score(test_X, testdata_Y))

# Iteration 3.3 - Using tfidftransformer with false smooth idf on the tfidfvectorized data
tfidf_transform = TfidfTransformer(smooth_idf=False)
tfidf_transform.fit(train_X)
train_X = tfidf_transform.transform(train_X)
test_X = tfidf_transform.transform(test_X)
log.fit(train_X, traindata_Y)
print("Iteration 3.3 F1 score: ", log.score(test_X, testdata_Y))

# Normalising the "Humor" data
traindata_Y = traindata_Y.replace("humor", "fake")

# Iteration 4 - Using CountVectorizer with 1000 features with specified evaluation metrics
bow = CountVectorizer(max_features=1000)
train_X = bow.fit_transform(traindata_X['tweetText'])
test_X = bow.fit_transform(testdata_X['tweetText'])
log.fit(train_X, traindata_Y)
log.score(test_X, testdata_Y)

# Finding the number of current True Negatives, False Positives, False Negatives and True Positives
tn_c, fp_c, fn_c, tp_c = confusion_matrix(testdata_Y, log.predict(test_X)).ravel()

print("---Iteration 4---")
print("Accuracy using the 'accuracy_score' function: ", accuracy_score(testdata_Y, log.predict(test_X)))


def evalmetrics(tn, fp, fn, tp):
    accuracy = (tp + tn)/(tp + fn + fp + tn)
    specificity = (tn)/(fp + tn)
    recall = (tp)/(tp + fn)
    precision = (tp)/(tp + fp)
    f1_score = (2 * precision * recall)/(precision + recall)
    fpr = (fp)/(fp + tn)
    auc = ((fpr * recall) / 2) + (recall * (1-fpr)) + (((1 - recall) * (1 - fpr)) / 2)
    print("Accuracy: ", accuracy)
    print("Specificity: ", specificity)
    print("Recall: ", recall)
    print("Precision: ", precision)
    print("F1-score: ", f1_score)
    print("AUC: ", auc)


print("Evaluation Metrics")
evalmetrics(tn_c, fp_c, fn_c, tp_c)


# Iteration 5 - Remove stop words
bow = CountVectorizer(max_features=1000, stop_words='english')
train_X = bow.fit_transform(traindata_X['tweetText'])
test_X = bow.fit_transform(testdata_X['tweetText'])
log.fit(train_X, traindata_Y)

tn_c, fp_c, fn_c, tp_c = confusion_matrix(testdata_Y, log.predict(test_X)).ravel()
print("Iteration 5 Evaluation Metrics")
evalmetrics(tn_c, fp_c, fn_c, tp_c)


# Iteration 6 - Changing fit_transform to transform for the test data
bow = CountVectorizer(max_features=1000)
train_X = bow.fit_transform(traindata_X['tweetText'])
test_X = bow.transform(testdata_X['tweetText'])
log.fit(train_X, traindata_Y)

tn_c, fp_c, fn_c, tp_c = confusion_matrix(testdata_Y, log.predict(test_X)).ravel()
print("Iteration 6 Evaluation Metrics")
evalmetrics(tn_c, fp_c, fn_c, tp_c)


# Iteration 7 - Changing fit_transform to transform for the test data and trying it with stop words
bow = CountVectorizer(max_features=1000, stop_words='english')
train_X = bow.fit_transform(traindata_X['tweetText'])
test_X = bow.transform(testdata_X['tweetText'])
log.fit(train_X, traindata_Y)

tn_c, fp_c, fn_c, tp_c = confusion_matrix(testdata_Y, log.predict(test_X)).ravel()
print("Iteration 7 Evaluation Metrics")
evalmetrics(tn_c, fp_c, fn_c, tp_c)


# Iteration 8 - Using the NLK stop words
bow = CountVectorizer(max_features=1000, stop_words=stopwords.words('english'))
train_X = bow.fit_transform(traindata_X['tweetText'])
test_X = bow.transform(testdata_X['tweetText'])
log.fit(train_X, traindata_Y)

tn_c, fp_c, fn_c, tp_c = confusion_matrix(testdata_Y, log.predict(test_X)).ravel()
print("Iteration 8 Evaluation Metrics")
evalmetrics(tn_c, fp_c, fn_c, tp_c)


def portStem(tweets):
    stemTweets = []
    for tweet in tweets:
        token_tweet = word_tokenize(tweet)
        stem_tweet = []
        for word in token_tweet:
            stem_tweet.append(PorterStemmer().stem(word))
            stem_tweet.append(" ")
        stemTweets.append("".join(stem_tweet))
    return pd.Series(stemTweets)


# Iteration 9 - Up the max features to 10,000 with porter stemmer
bow = CountVectorizer(max_features=10000, stop_words=stopwords.words('english'))
# = is referencing the old list | .copy makes a whole different list of the original
train_X = bow.fit_transform(portStem(traindata_X['tweetText'].copy()))
test_X = bow.transform(portStem(testdata_X['tweetText'].copy()))
log.fit(train_X, traindata_Y)

tn_c, fp_c, fn_c, tp_c = confusion_matrix(testdata_Y, log.predict(test_X)).ravel()
print("Iteration 9 Evaluation Metrics")
evalmetrics(tn_c, fp_c, fn_c, tp_c)


# Iteration 10 - Up the max features to 10,000 without the porter stemmer
bow = CountVectorizer(max_features=10000, stop_words=stopwords.words('english'))
train_X = bow.fit_transform(traindata_X['tweetText'])
test_X = bow.transform(testdata_X['tweetText'])
log.fit(train_X, traindata_Y)

tn_c, fp_c, fn_c, tp_c = confusion_matrix(testdata_Y, log.predict(test_X)).ravel()
print("Iteration 10 Evaluation Metrics")
evalmetrics(tn_c, fp_c, fn_c, tp_c)


# Iteration 11 - Features at 1000 with the porter stemmer
bow = CountVectorizer(max_features=1000, stop_words=stopwords.words('english'))
train_X = bow.fit_transform(portStem(traindata_X['tweetText'].copy()))
test_X = bow.transform(portStem(testdata_X['tweetText'].copy()))
log.fit(train_X, traindata_Y)

tn_c, fp_c, fn_c, tp_c = confusion_matrix(testdata_Y, log.predict(test_X)).ravel()
print("Iteration 11 Evaluation Metrics")
evalmetrics(tn_c, fp_c, fn_c, tp_c)


def lancastStem(tweets):
    stemTweets = []
    for tweet in tweets:
        token_tweet = word_tokenize(tweet)
        stem_tweet = []
        for word in token_tweet:
            stem_tweet.append(LancasterStemmer().stem(word))
            stem_tweet.append(" ")
        stemTweets.append("".join(stem_tweet))
    return pd.Series(stemTweets)


# Iteration 12 - Return to 10,000 features with the lancaster stemmer
bow = CountVectorizer(max_features=10000, stop_words=stopwords.words('english'))
train_X = bow.fit_transform(lancastStem(traindata_X['tweetText'].copy()))
test_X = bow.transform(lancastStem(testdata_X['tweetText'].copy()))
log.fit(train_X, traindata_Y)

tn_c, fp_c, fn_c, tp_c = confusion_matrix(testdata_Y, log.predict(test_X)).ravel()
print("Iteration 12 Evaluation Metrics")
evalmetrics(tn_c, fp_c, fn_c, tp_c)


def wordNetLem(tweets):
    lemTweets = []
    for tweet in tweets:
        token_tweet = word_tokenize(tweet)
        lem_tweet = []
        for word in token_tweet:
            lem_tweet.append(WordNetLemmatizer().lemmatize(word))
            lem_tweet.append(" ")  # see if you can do a better spaces
        lemTweets.append("".join(lem_tweet))
    return pd.Series(lemTweets)


# Iteration 13 - Using wordnet lemmatizer
bow = CountVectorizer(max_features=10000, stop_words=stopwords.words('english'))
train_X = bow.fit_transform(wordNetLem(traindata_X['tweetText'].copy()))
test_X = bow.transform(wordNetLem(testdata_X['tweetText'].copy()))
log.fit(train_X, traindata_Y)

tn_c, fp_c, fn_c, tp_c = confusion_matrix(testdata_Y, log.predict(test_X)).ravel()
print("Iteration 13 Evaluation Metrics")
evalmetrics(tn_c, fp_c, fn_c, tp_c)


def wordNetLemV(tweets):
    lemTweets = []
    for tweet in tweets:
        token_tweet = word_tokenize(tweet)
        lem_tweet = []
        for word in token_tweet:
            lem_tweet.append(WordNetLemmatizer().lemmatize(word, pos="v"))
            lem_tweet.append(" ")  # see if you can do a better spaces
        lemTweets.append("".join(lem_tweet))
    return pd.Series(lemTweets)


# Iteration 14 - Using wordnet lemmatizer with POS
bow = CountVectorizer(max_features=10000, stop_words=stopwords.words('english'))
train_X = bow.fit_transform(wordNetLemV(traindata_X['tweetText'].copy()))
test_X = bow.transform(wordNetLemV(testdata_X['tweetText'].copy()))
log.fit(train_X, traindata_Y)

tn_c, fp_c, fn_c, tp_c = confusion_matrix(testdata_Y, log.predict(test_X)).ravel()
print("Iteration 14 Evaluation Metrics")
evalmetrics(tn_c, fp_c, fn_c, tp_c)


# Iteration 15 - Lemmaizing, and then stemming
train_X = bow.fit_transform(portStem(wordNetLemV(traindata_X['tweetText'].copy())))
test_X = bow.transform(portStem(wordNetLemV(testdata_X['tweetText'].copy())))
log.fit(train_X, traindata_Y)

tn_c, fp_c, fn_c, tp_c = confusion_matrix(testdata_Y, log.predict(test_X)).ravel()
print("Iteration 15 Evaluation Metrics")
evalmetrics(tn_c, fp_c, fn_c, tp_c)


# Iteration 16 - Stemming and then lemmaizing
train_X = bow.fit_transform(wordNetLemV(portStem(traindata_X['tweetText'].copy())))
test_X = bow.transform(wordNetLemV(portStem(testdata_X['tweetText'].copy())))
log.fit(train_X, traindata_Y)

tn_c, fp_c, fn_c, tp_c = confusion_matrix(testdata_Y, log.predict(test_X)).ravel()
print("Iteration 16 Evaluation Metrics")
evalmetrics(tn_c, fp_c, fn_c, tp_c)


# Iteration 17 - Use lemmaization with TFIDF
tfidf = TfidfVectorizer(max_features=10000, stop_words=stopwords.words('english'))
train_X = tfidf.fit_transform(wordNetLemV(traindata_X['tweetText'].copy()))
test_X = tfidf.transform(wordNetLem(testdata_X['tweetText'].copy()))
log.fit(train_X, traindata_Y)

tn_c, fp_c, fn_c, tp_c = confusion_matrix(testdata_Y, log.predict(test_X)).ravel()
print("Iteration 17 Evaluation Metrics")
evalmetrics(tn_c, fp_c, fn_c, tp_c)


# Iteration 18 - Use lemmaization with TFIDFTransform
tfidf = TfidfVectorizer(max_features=10000, stop_words=stopwords.words('english'))
train_X = tfidf.fit_transform(wordNetLemV(traindata_X['tweetText'].copy()))
test_X = tfidf.transform(wordNetLem(testdata_X['tweetText'].copy()))
tfidf_transform = TfidfTransformer()
tfidf_transform.fit(train_X)
train_X = tfidf_transform.transform(train_X)
test_X = tfidf_transform.transform(test_X)
log.fit(train_X, traindata_Y)

tn_c, fp_c, fn_c, tp_c = confusion_matrix(testdata_Y, log.predict(test_X)).ravel()
print("Iteration 18 Evaluation Metrics")
evalmetrics(tn_c, fp_c, fn_c, tp_c)


# Iteration 19 - Use stemming with TFIDF
tfidf = TfidfVectorizer(max_features=10000, stop_words=stopwords.words('english'))
train_X = tfidf.fit_transform(portStem(traindata_X['tweetText'].copy()))
test_X = tfidf.transform(portStem(testdata_X['tweetText'].copy()))
log.fit(train_X, traindata_Y)

tn_c, fp_c, fn_c, tp_c = confusion_matrix(testdata_Y, log.predict(test_X)).ravel()
print("Iteration 19 Evaluation Metrics")
evalmetrics(tn_c, fp_c, fn_c, tp_c)


# Iteration 20 - Removing most of the URL with only lemmization
def removeURL(tweets):
    removedurls = []
    for tweet in tweets:
        removedurls.append(tweet.replace("http://t.co/",""))
    return pd.Series(removedurls)


tfidf = TfidfVectorizer(max_features=10000, stop_words=stopwords.words('english'))
cleantraindata_X = removeURL(traindata_X['tweetText'].copy())
cleantestdata_X = removeURL(testdata_X['tweetText'].copy())
train_X = tfidf.fit_transform(wordNetLemV(cleantraindata_X))
test_X = tfidf.transform(wordNetLem(cleantestdata_X))
tfidf_transform = TfidfTransformer()
tfidf_transform.fit(train_X)
train_X = tfidf_transform.transform(train_X)
test_X = tfidf_transform.transform(test_X)
log.fit(train_X, traindata_Y)

tn_c, fp_c, fn_c, tp_c = confusion_matrix(testdata_Y, log.predict(test_X)).ravel()
print("Iteration 20 Evaluation Metrics")  # Values are the same as iteration 18
evalmetrics(tn_c, fp_c, fn_c, tp_c)


from sklearn.linear_model import LogisticRegressionCV

# Iteration 21 - Trying logCV
logCV = LogisticRegressionCV(Cs=10)
tfidf = TfidfVectorizer(max_features=10000, stop_words=stopwords.words('english'))
train_X = tfidf.fit_transform(wordNetLemV(traindata_X['tweetText'].copy()))
test_X = tfidf.transform(wordNetLem(testdata_X['tweetText'].copy()))
tfidf_transform = TfidfTransformer()
tfidf_transform.fit(train_X)
train_X = tfidf_transform.transform(train_X)
test_X = tfidf_transform.transform(test_X)
logCV.fit(train_X, traindata_Y)

tn_c, fp_c, fn_c, tp_c = confusion_matrix(testdata_Y, log.predict(test_X)).ravel()
print("Iteration 21 Evaluation Metrics")
evalmetrics(tn_c, fp_c, fn_c, tp_c)


def datasplit5050(data):
    data['label'] = data['label'].replace("humor", "fake")
    realdata = np.array_split(data[data['label'] == 'real'], 2)
    fakedata = np.array_split(data[data['label'] == 'fake'], 2)
    train50 = shuffle(pd.concat([realdata[0], fakedata[0]]), random_state=0)
    valid50 = shuffle(pd.concat([realdata[1], fakedata[1]]), random_state=0)
    return train50, valid50


def datasplit8020(data):
    data['label'] = data['label'].replace("humor", "fake")
    realdata = data[data['label'] == 'real']
    fakedata = data[data['label'] == 'fake']
    realindices = round(realdata.shape[0] * 0.8)
    fakeindices = round(fakedata.shape[0] * 0.8)
    train80 = shuffle(pd.concat([realdata[:realindices], fakedata[:fakeindices]]), random_state=0)
    valid20 = shuffle(pd.concat([realdata[realindices:], fakedata[fakeindices:]]), random_state=0)
    return train80, valid20


def datasplit7525(data):
    data['label'] = data['label'].replace("humor", "fake")
    realdata = data[data['label'] == 'real']
    fakedata = data[data['label'] == 'fake']
    realindices = round(realdata.shape[0] * 0.75)
    fakeindices = round(fakedata.shape[0] * 0.75)
    train80 = shuffle(pd.concat([realdata[:realindices], fakedata[:fakeindices]]), random_state=0)
    valid20 = shuffle(pd.concat([realdata[realindices:], fakedata[fakeindices:]]), random_state=0)
    return train80, valid20


def crossval10(data):
    data['label'] = data['label'].replace("humor", "fake")
    realdata = np.array_split(data[data['label'] == 'real'], 10)
    fakedata = np.array_split(data[data['label'] == 'fake'], 10)
    kfolds = []
    for real, fake in zip(realdata, fakedata):
        kfolds.append(shuffle(pd.concat([real, fake])))
    return kfolds


def testOne(validationtech, logreg, train, validate, test, tfidfVect, tfidfTrans, countVect):
    print(validationtech)
    trainY = train.loc[:, 'label']
    validateY = validate.loc[:, 'label']
    testY = test.loc[:, 'label']

    print("\nTF-IDF")
    trainX = train.loc[:, ['tweetText']]
    trainX = wordNetLemV(trainX['tweetText'])
    trainX = tfidfVect.fit_transform(trainX)
    tfidfTrans.fit(trainX)
    trainX = tfidfTrans.transform(trainX)
    validateX = validate.loc[:, ['tweetText']]
    validateX = wordNetLemV(validateX['tweetText'])
    validateX = tfidfTrans.transform(tfidfVect.transform(validateX))
    testX = test.loc[:, ['tweetText']]
    testX = wordNetLemV(testX['tweetText'])
    testX = tfidfTrans.transform(tfidfVect.transform(testX))
    logreg.fit(trainX, trainY)
    vtn, vfp, vfn, vtp = confusion_matrix(validateY, logreg.predict(validateX)).ravel()
    ttn, tfp, tfn, ttp = confusion_matrix(testY, logreg.predict(testX)).ravel()
    print("\nValidation Set Scores")
    evalmetrics(vtn, vfp, vfn, vtp)
    print("\nTest Set Scores")
    evalmetrics(ttn, tfp, tfn, ttp)
    print("\n")

    print("Bag-of-Words")
    trainX = train.loc[:, ['tweetText']]
    trainX = wordNetLemV(trainX['tweetText'])
    trainX = countVect.fit_transform(trainX)
    validateX = validate.loc[:, ['tweetText']]
    validateX = wordNetLemV(validateX['tweetText'])
    validateX = countVect.transform(validateX)
    testX = test.loc[:, ['tweetText']]
    testX = wordNetLemV(testX['tweetText'])
    testX = countVect.transform(testX)
    logreg.fit(trainX, trainY)
    vtn, vfp, vfn, vtp = confusion_matrix(validateY, logreg.predict(validateX)).ravel()
    ttn, tfp, tfn, ttp = confusion_matrix(testY, logreg.predict(testX)).ravel()
    print("\nValidation Set Scores")
    evalmetrics(vtn, vfp, vfn, vtp)
    print("\nTest Set Scores")
    evalmetrics(ttn, tfp, tfn, ttp)
    print("\n")


def testAll(logreg, data, datatest, tfidfVect, tfidfTrans, countVect):
    train, validate = datasplit5050(data)
    testOne("Train Data Split 50-50", logreg, train, validate, datatest, tfidfVect, tfidfTrans, countVect)

    train, validate = datasplit8020(data)
    testOne("Train Data Split 80-20", logreg, train, validate, datatest, tfidfVect, tfidfTrans, countVect)

    train, validate = datasplit7525(data)
    testOne("Train Data Split 75-25", logreg, train, validate, datatest, tfidfVect, tfidfTrans, countVect)


# Iteration 22 - Trying validation data set, data splits of 50|50, 80|20, 75|25
print("Iteration 22 Evaluation Metrics")
testAll(LogisticRegression(), traindata.loc[:, ['tweetText', 'label']], testdata.loc[:, ['tweetText', 'label']],
        TfidfVectorizer(max_features=10000, stop_words=stopwords.words('english')),
        TfidfTransformer(),
        CountVectorizer(max_features=10000, stop_words=stopwords.words('english')))


# Iteration 23 - Attempt with 100,000 features | Result: Decided 50|50 split isn't good
testAll(LogisticRegression(), traindata.loc[:, ['tweetText', 'label']] , testdata.loc[:, ['tweetText', 'label']],
        TfidfVectorizer(max_features=100000, stop_words=stopwords.words('english')),
        TfidfTransformer(),
        CountVectorizer(max_features=100000, stop_words=stopwords.words('english')))


# Iteration 24 - 50000 features
logreg = LogisticRegression()
data = traindata.loc[:, ['tweetText', 'label']]
datatest = testdata.loc[:, ['tweetText', 'label']]
tfidfVect = TfidfVectorizer(max_features=50000, stop_words=stopwords.words('english'))
tfidfTrans = TfidfTransformer()
countVect = CountVectorizer(max_features=50000, stop_words=stopwords.words('english'))

train, validate = datasplit8020(data)
testOne("Train Data Split 80-20", logreg, train, validate, datatest, tfidfVect, tfidfTrans, countVect)

train, validate = datasplit7525(data)
testOne("Train Data Split 75-25", logreg, train, validate, datatest, tfidfVect, tfidfTrans, countVect)


# Iteration 25 - 25000 features
logreg = LogisticRegression()
data = traindata.loc[:, ['tweetText', 'label']]
datatest = testdata.loc[:, ['tweetText', 'label']]
tfidfVect = TfidfVectorizer(max_features=25000, stop_words=stopwords.words('english'))
tfidfTrans = TfidfTransformer()
countVect = CountVectorizer(max_features=25000, stop_words=stopwords.words('english'))

train, validate = datasplit8020(data)
testOne("Train Data Split 80-20", logreg, train, validate, datatest, tfidfVect, tfidfTrans, countVect)

train, validate = datasplit7525(data)
testOne("Train Data Split 75-25", logreg, train, validate, datatest, tfidfVect, tfidfTrans, countVect)


# Iteration 26 -  back to 10000 features
logreg = LogisticRegression()
data = traindata.loc[:, ['tweetText', 'label']]
datatest = testdata.loc[:, ['tweetText', 'label']]
tfidfVect = TfidfVectorizer(max_features=10000, stop_words=stopwords.words('english'))
tfidfTrans = TfidfTransformer()
countVect = CountVectorizer(max_features=10000, stop_words=stopwords.words('english'))

train, validate = datasplit8020(data)
testOne("Train Data Split 80-20", logreg, train, validate, datatest, tfidfVect, tfidfTrans, countVect)

train, validate = datasplit7525(data)
testOne("Train Data Split 75-25", logreg, train, validate, datatest, tfidfVect, tfidfTrans, countVect)


# Iteration 27 -  using bigrams + unigrams too
logreg = LogisticRegression()
data = traindata.loc[:, ['tweetText', 'label']]
datatest = testdata.loc[:, ['tweetText', 'label']]
tfidfVect = TfidfVectorizer(max_features=10000, stop_words=stopwords.words('english'), ngram_range=(1, 2))
tfidfTrans = TfidfTransformer()
countVect = CountVectorizer(max_features=10000, stop_words=stopwords.words('english'), ngram_range=(1, 2))

train, validate = datasplit8020(data)
testOne("Train Data Split 80-20", logreg, train, validate, datatest, tfidfVect, tfidfTrans, countVect)

train, validate = datasplit7525(data)
testOne("Train Data Split 75-25", logreg, train, validate, datatest, tfidfVect, tfidfTrans, countVect)


# Iteration 28 -  using bigrams + unigrams + trigrams too
logreg = LogisticRegression()
data = traindata.loc[:, ['tweetText', 'label']]
datatest = testdata.loc[:, ['tweetText', 'label']]
tfidfVect = TfidfVectorizer(max_features=10000, stop_words=stopwords.words('english'), ngram_range=(1, 3))
tfidfTrans = TfidfTransformer()
countVect = CountVectorizer(max_features=10000, stop_words=stopwords.words('english'), ngram_range=(1, 3))

train, validate = datasplit8020(data)
testOne("Train Data Split 80-20", logreg, train, validate, datatest, tfidfVect, tfidfTrans, countVect)

train, validate = datasplit7525(data)
testOne("Train Data Split 75-25", logreg, train, validate, datatest, tfidfVect, tfidfTrans, countVect)


# Iteration 29 -  using bigrams + trigrams
logreg = LogisticRegression()
data = traindata.loc[:, ['tweetText', 'label']]
datatest = testdata.loc[:, ['tweetText', 'label']]
tfidfVect = TfidfVectorizer(max_features=10000, stop_words=stopwords.words('english'), ngram_range=(2, 3))
tfidfTrans = TfidfTransformer()
countVect = CountVectorizer(max_features=10000, stop_words=stopwords.words('english'), ngram_range=(2, 3))

train, validate = datasplit8020(data)
testOne("Train Data Split 80-20", logreg, train, validate, datatest, tfidfVect, tfidfTrans, countVect)

train, validate = datasplit7525(data)
testOne("Train Data Split 75-25", logreg, train, validate, datatest, tfidfVect, tfidfTrans, countVect)


from scipy.sparse import csr_matrix, hstack


def testTextUsernameOne(validationtech, logreg, train, validate, test, tfidfVect, tfidfTrans, countVect):
    print(validationtech, " (TweetText and Username)")
    trainY = train.loc[:, 'label']
    validateY = validate.loc[:, 'label']
    testY = test.loc[:, 'label']

    print("\nTF-IDF")
    trainX = train.loc[:, ['tweetText']]
    trainX = wordNetLemV(trainX['tweetText'])
    tfidfTrans.fit(tfidfVect.fit_transform(trainX))
    trainX = tfidfTrans.transform(tfidfVect.fit_transform(trainX))
    trainX = hstack((trainX, csr_matrix(((train.loc[:, ['username']]).values).astype(np.int))))

    validateX = validate.loc[:, ['tweetText']]
    validateX = wordNetLemV(validateX['tweetText'])
    validateX = tfidfTrans.transform(tfidfVect.transform(validateX))
    validateX = hstack((validateX, csr_matrix(((validate.loc[:, ['username']]).values).astype(np.int))))

    testX = test.loc[:, ['tweetText']]
    testX = wordNetLemV(testX['tweetText'])
    testX = tfidfTrans.transform(tfidfVect.transform(testX))
    testX = hstack((testX, csr_matrix(((test.loc[:, ['username']]).values).astype(np.int))))

    logreg.fit(trainX, trainY)
    vtn, vfp, vfn, vtp = confusion_matrix(validateY, logreg.predict(validateX)).ravel()
    ttn, tfp, tfn, ttp = confusion_matrix(testY, logreg.predict(testX)).ravel()
    print("\nValidation Set Scores")
    evalmetrics(vtn, vfp, vfn, vtp)
    print("\nTest Set Scores")
    evalmetrics(ttn, tfp, tfn, ttp)
    print("\n")

    print("Bag-of-Words")
    trainX = train.loc[:, ['tweetText']]
    trainX = wordNetLemV(trainX['tweetText'])
    trainX = countVect.fit_transform(trainX)
    trainX = hstack((trainX, csr_matrix(((train.loc[:, ['username']]).values).astype(np.int))))

    validateX = validate.loc[:, ['tweetText']]
    validateX = wordNetLemV(validateX['tweetText'])
    validateX = countVect.transform(validateX)
    validateX = hstack((validateX, csr_matrix(((validate.loc[:, ['username']]).values).astype(np.int))))

    testX = test.loc[:, ['tweetText']]
    testX = wordNetLemV(testX['tweetText'])
    testX = countVect.transform(testX)
    testX = hstack((testX, csr_matrix(((test.loc[:, ['username']]).values).astype(np.int))))

    logreg.fit(trainX, trainY)
    vtn, vfp, vfn, vtp = confusion_matrix(validateY, logreg.predict(validateX)).ravel()
    ttn, tfp, tfn, ttp = confusion_matrix(testY, logreg.predict(testX)).ravel()
    print("\nValidation Set Scores")
    evalmetrics(vtn, vfp, vfn, vtp)
    print("\nTest Set Scores")
    evalmetrics(ttn, tfp, tfn, ttp)
    print("\n")


# Iteration 30 - Using length of username as a feature | Result: Makes completely no difference
def lengths(data):
    for i, username in enumerate(data['username']):
        username = len(username)
        data['username'][i] = username
    return data


logreg = LogisticRegression()
data = lengths(traindata.loc[:, ['tweetText', 'username', 'label']])
datatest = lengths(testdata.loc[:, ['tweetText', 'username', 'label']])
tfidfVect = TfidfVectorizer(max_features=10000, stop_words=stopwords.words('english'), ngram_range=(1, 3))
tfidfTrans = TfidfTransformer()

train, validate = datasplit8020(data)
testTextUsernameOne("Train Data Split 80-20", logreg, train, validate, datatest, tfidfVect, tfidfTrans, countVect)


# Iteration 31 - Fine-tuning Log regression through validation datasets
from sklearn.model_selection import GridSearchCV

logreg = LogisticRegression()
grid = {'penalty': ['l1', 'l2', 'elasticnet', 'none'],
        'tol': [0.0001, 0.0005],
        'C': [1.0, 0.5],
        'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
        'max_iter': [100, 200, 300, 400, 500],
        'multi_class': ['auto', 'ovr'],
        'n_jobs': [-1]}

logreg_cv = GridSearchCV(estimator=logreg, param_grid=grid, scoring='f1_micro', cv=10, n_jobs=-1)


def fineTune(validationtech, gridlogreg, train, test, tfidfVect, tfidfTrans, countVect):
    print(validationtech)
    trainY = train.loc[:, 'label']
    validateY = validate.loc[:, 'label']
    testY = test.loc[:, 'label']

    print("\nTF-IDF")
    trainX = train.loc[:, ['tweetText']]
    trainX = wordNetLemV(trainX['tweetText'])
    trainX = tfidfVect.fit_transform(trainX)
    tfidfTrans.fit(trainX)
    trainX = tfidfTrans.transform(trainX)
    testX = test.loc[:, ['tweetText']]
    testX = wordNetLemV(testX['tweetText'])
    testX = tfidfTrans.transform(tfidfVect.transform(testX))
    gridlogreg.fit(trainX, trainY)

    print("Best score: ", gridlogreg.best_score_)
    print("Best Parameters: ", gridlogreg.best_params_)

    ttn, tfp, tfn, ttp = confusion_matrix(testY, gridlogreg.predict(testX)).ravel()
    print("\nTest Set Scores")
    evalmetrics(ttn, tfp, tfn, ttp)
    print("\n")

    print("Bag-of-Words")
    trainX = train.loc[:, ['tweetText']]
    trainX = wordNetLemV(trainX['tweetText'])
    trainX = countVect.fit_transform(trainX)
    testX = test.loc[:, ['tweetText']]
    testX = wordNetLemV(testX['tweetText'])
    testX = countVect.transform(testX)
    gridlogreg.fit(trainX, trainY)
    print("Best score: ", gridlogreg.best_score_)
    print("Best Parameters: ", gridlogreg.best_params_)

    ttn, tfp, tfn, ttp = confusion_matrix(testY, gridlogreg.predict(testX)).ravel()
    print("\nTest Set Scores")
    evalmetrics(ttn, tfp, tfn, ttp)
    print("\n")


data = traindata.loc[:, ['tweetText', 'label']]
datatest = testdata.loc[:, ['tweetText', 'label']]
tfidfVect = TfidfVectorizer(max_features=10000, stop_words=stopwords.words('english'), ngram_range=(1, 3))
tfidfTrans = TfidfTransformer()
countVect = CountVectorizer(max_features=10000, stop_words=stopwords.words('english'), ngram_range=(1, 3))

fineTune("Train Data Split 80-20", logreg_cv, data, datatest, tfidfVect, tfidfTrans, countVect)


# Iteration 31.1 -  Using the initial "best" parameters
logreg = LogisticRegression(C=1.0, max_iter=200, multi_class='auto', n_jobs=-1, penalty='l2', solver='lbfgs', tol=0.0001)
data = traindata.loc[:, ['tweetText', 'label']]
datatest = testdata.loc[:, ['tweetText', 'label']]
tfidfVect = TfidfVectorizer(max_features=10000, stop_words=stopwords.words('english'), ngram_range=(1, 3))
tfidfTrans = TfidfTransformer()
countVect = CountVectorizer(max_features=10000, stop_words=stopwords.words('english'), ngram_range=(1, 3))

train, validate = datasplit8020(data)
testOne("Train Data Split 80-20", logreg, train, validate, datatest, tfidfVect, tfidfTrans, countVect)

# Train Data Split 80-20
#


# Iteration 32 - Different fine-tuning
logreg = LogisticRegression()
grid = {'penalty': ['l2'],
        'tol': [0.0001, 0.0002],
        'C': [1.0, 0.9, 0.8],
        'solver': ['lbfgs'],
        'max_iter': [100, 150, 200, 250, 300],
        'multi_class': ['auto'],
        'n_jobs': [-1]}

logreg_cv = GridSearchCV(estimator=logreg, param_grid=grid, scoring='f1_micro', cv=10, n_jobs=-1)

data = traindata.loc[:, ['tweetText', 'label']]
datatest = testdata.loc[:, ['tweetText', 'label']]
tfidfVect = TfidfVectorizer(max_features=10000, stop_words=stopwords.words('english'), ngram_range=(1, 3))
tfidfTrans = TfidfTransformer()
countVect = CountVectorizer(max_features=10000, stop_words=stopwords.words('english'), ngram_range=(1, 3))

fineTune("Train Data Split 80-20", logreg_cv, data, datatest, tfidfVect, tfidfTrans, countVect)


# Iteration 32.5 -  Using the initial "best" parameters for lbfgs
logreg = LogisticRegression(C=0.8, max_iter=150, multi_class='auto', n_jobs=-1, penalty='l2', solver='lbfgs', tol=0.0001)
data = traindata.loc[:, ['tweetText', 'label']]
datatest = testdata.loc[:, ['tweetText', 'label']]
tfidfVect = TfidfVectorizer(max_features=10000, stop_words=stopwords.words('english'), ngram_range=(1, 3))
tfidfTrans = TfidfTransformer()
countVect = CountVectorizer(max_features=10000, stop_words=stopwords.words('english'), ngram_range=(1, 3))

train, validate = datasplit8020(data)
testOne("Train Data Split 80-20", logreg, train, validate, datatest, tfidfVect, tfidfTrans, countVect)


# Iteration 33 - Using the full dataset to train and then using the test set only
logreg = LogisticRegression(C=0.8, max_iter=150, multi_class='auto', n_jobs=-1, penalty='l2', solver='lbfgs', tol=0.0001)
data = traindata.loc[:, ['tweetText', 'label']]
datatest = testdata.loc[:, ['tweetText', 'label']]
tfidfVect = TfidfVectorizer(max_features=10000, stop_words=stopwords.words('english'), ngram_range=(1, 3))
tfidfTrans = TfidfTransformer()
countVect = CountVectorizer(max_features=10000, stop_words=stopwords.words('english'), ngram_range=(1, 3))

testOne("Train Data Split 80-20", logreg, data, datatest, datatest, tfidfVect, tfidfTrans, countVect)


# Iteration 34 - with the built-in stop words + Using the full dataset to train and then using the test set only
logreg = LogisticRegression(C=0.8, max_iter=150, multi_class='auto', n_jobs=-1, penalty='l2', solver='lbfgs', tol=0.0001)
data = traindata.loc[:, ['tweetText', 'label']]
datatest = testdata.loc[:, ['tweetText', 'label']]
tfidfVect = TfidfVectorizer(max_features=10000, stop_words='english', ngram_range=(1, 3))
tfidfTrans = TfidfTransformer()
countVect = CountVectorizer(max_features=10000, stop_words='english', ngram_range=(1, 3))

testOne("Train Data Split 80-20", logreg, data, datatest, datatest, tfidfVect, tfidfTrans, countVect)
