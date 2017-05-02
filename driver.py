import re
import string
import os
import pandas as pd
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import SGDClassifier


train_path = "../aclImdb/train/" # source data
test_path = "../imdb_te.csv" # test data for grade evaluation. 


def imdb_data_preprocess(inpath, outpath="./", name="imdb_tr.csv", mix=False):
        pos_path = inpath + "pos/"
        neg_path = inpath + "neg/"
        row = 0
        rows = []
        contents = []
        polarity = []
        regex1 = re.compile('<.*?>')
        regex2 = re.compile('[%s0-9]' % re.escape(string.punctuation))
        regex3 = re.compile('[^\x00-\x7F]')
        sw = open("../stopwords.en.txt", "r")

        #get rid of the last element which is space
        stopwords = set(sw.read().split('\n')[:-1])
        sw.close()

        for filename in os.listdir(pos_path):
                filepath = pos_path+filename
                inF = open(filepath, "r")
                content = inF.read().lower()
                inF.close()
                content = regex1.sub(' ',content)
                content = regex2.sub(' ',content)
                content = regex3.sub(' ',content)

                content_list = content.split()
                content = ' '.join(i for i in content_list if i not in stopwords)
                
                rows.append(row)
                contents.append(content)
                polarity.append(1)
                row += 1

        dataSet = list(zip(rows,contents,polarity))
        df = pd.DataFrame(data = dataSet, columns = ['row_number','text','polarity'])
        df.to_csv(outpath+name, index=False, header=True)
        rows = []
        contents = []
        polarity = []

        for filename in os.listdir(neg_path):
                filepath = neg_path+filename
                inF = open(filepath, "r")
                content = inF.read().lower()
                inF.close()
                content = regex1.sub(' ',content)
                content = regex2.sub(' ',content)
                content = regex3.sub(' ',content)

                content_list = content.split()
                content = ' '.join(i for i in content_list if i not in stopwords)
                
                rows.append(row)
                contents.append(content)
                polarity.append(0)
                row += 1

        dataSet = list(zip(rows,contents,polarity))
        df = pd.DataFrame(data = dataSet, columns = ['row_number','text','polarity'])
        df.to_csv(outpath+name, index=False, header=False, mode='a')
  
if __name__ == "__main__":
        #imdb_data_preprocess(train_path)
        
        #df_train = pd.read_csv('./imdb_tr.csv')
        
        X = []
        Y_train = []
        pos_path = train_path + "pos/"
        neg_path = train_path + "neg/"
        regex1 = re.compile('<.*?>')
        regex2 = re.compile('[%s0-9]' % re.escape(string.punctuation))
        regex3 = re.compile('[^\x00-\x7F]')
        sw = open("../stopwords.en.txt", "r")

        #get rid of the last element which is space
        stopwords = sw.read().split('\n')[:-1]
        sw.close()

        for filename in os.listdir(pos_path):
                filepath = pos_path+filename
                inF = open(filepath, "r")
                content = inF.read().lower()
                inF.close()
                content = regex1.sub(' ',content)
                content = regex2.sub(' ',content)
                content = regex3.sub(' ',content)

                X.append(content)
                Y_train.append(1)

        for filename in os.listdir(neg_path):
                filepath = neg_path+filename
                inF = open(filepath, "r")
                content = inF.read().lower()
                inF.close()
                content = regex1.sub(' ',content)
                content = regex2.sub(' ',content)
                content = regex3.sub(' ',content)

                X.append(content)
                Y_train.append(0)

        X_test = []
        df_test = pd.read_csv(test_path, encoding = "ISO-8859-1")
        for index, row in df_test.iterrows():
                content = row['text'].lower()
                content = regex1.sub(' ',content)
                content = regex2.sub(' ',content)
                content = regex3.sub(' ',content)
                X_test.append(content)

        outF1 = open("unigram.output.txt", "w")
        outF2 = open("unigramtfidf.output.txt", "w")
        outF3 = open("bigram.output.txt", "w")
        outF4 = open("bigramtfidf.output.txt", "w")

        """ unigram  """
        unigram = CountVectorizer(stop_words=stopwords)
        X_train_unigram = unigram.fit_transform(X)
        X_test_unigram = unigram.transform(X_test)
        
        sgd1 = SGDClassifier(penalty='l1')
        sgd1.fit(X_train_unigram, Y_train)
        Y_test = sgd1.predict(X_test_unigram)
        for result in Y_test:
                outF1.write(str(result))
                outF1.write("\n")

        outF1.close()

        """ unigram with tfidf  """
        tfidf_unigram = TfidfVectorizer(stop_words=stopwords)
        X_train_tf_unigram = tfidf_unigram.fit_transform(X)
        X_test_tf_unigram = tfidf_unigram.transform(X_test)

        sgd2 = SGDClassifier(penalty='l1')
        sgd2.fit(X_train_tf_unigram, Y_train)
        Y_test = sgd2.predict(X_test_tf_unigram)
        for result in Y_test:
                outF2.write(str(result))
                outF2.write("\n")

        outF2.close()

        """ bigram  """
        bigram = CountVectorizer(ngram_range=(2,2),stop_words=stopwords)
        X_train_bigram = bigram.fit_transform(X)
        X_test_bigram = bigram.transform(X_test)

        sgd3 = SGDClassifier(penalty='l1')
        sgd3.fit(X_train_bigram, Y_train)
        Y_test = sgd3.predict(X_test_bigram)
        for result in Y_test:
                outF3.write(str(result))
                outF3.write("\n")

        outF3.close()

        
        """ bigram with tfidf  """
        tfidf_bigram = TfidfVectorizer(ngram_range=(2,2),stop_words=stopwords)
        X_train_tf_bigram = tfidf_bigram.fit_transform(X)
        X_test_tf_bigram = tfidf_bigram.transform(X_test)

        sgd4 = SGDClassifier(penalty='l1')
        sgd4.fit(X_train_tf_bigram, Y_train)
        Y_test = sgd4.predict(X_test_tf_bigram)
        for result in Y_test:
                outF4.write(str(result))
                outF4.write("\n")

        outF4.close()
        
        
        
        #clf = GridSearchCV(sgd,param,cv=5)
        #clf.fit(X_train, Y_train)

        """
        print(cross_val_score(sgd, X_train_unigram, Y_train, cv=5))
        print(cross_val_score(sgd, X_train_bigram, Y_train, cv=5))
        print(cross_val_score(sgd, X_train_tf_unigram, Y_train, cv=5))
        print(cross_val_score(sgd, X_train_tf_bigram, Y_train, cv=5))
        """
        
        #print(clf.cv_results_['mean_train_score'])
        #clf.predict 
        
