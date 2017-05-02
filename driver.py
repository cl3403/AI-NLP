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
        
        df_train = pd.read_csv('./imdb_tr.csv')
        #df_test = pd.read_csv('../imdb_te.csv', encoding = "ISO-8859-1")

        X = []
        Y_train = []
        for index, row in df_train.iterrows():
                X.append(row['text'])
                Y_train.append(row['polarity'])

        #for index, row in df_test.iterrows
                 
        sw = open("../stopwords.en.txt", "r")

        #get rid of the last element which is space
        stopwords = sw.read().split('\n')[:-1]
        sw.close()
        unigram = CountVectorizer(stop_words=stopwords)
        X_train_unigram = unigram.fit_transform(X)
        #param = {'loss':['hinge'],'penalty':['l1']}

        bigram = CountVectorizer(ngram_range=(2,2),stop_words=stopwords)
        X_train_bigram = bigram.fit_transform(X)

        tfidf_unigram = TfidfVectorizer(stop_words=stopwords)
        X_train_tf_unigram = tfidf_unigram.fit_transform(X)
        
        tfidf_bigram = TfidfVectorizer(ngram_range=(2,2),stop_words=stopwords)
        X_train_tf_bigram = tfidf_bigram.fit_transform(X)
        
        sgd = SGDClassifier(penalty='l1')
        #clf = GridSearchCV(sgd,param,cv=5)
        #clf.fit(X_train, Y_train)

        
        print(cross_val_score(sgd, X_train_unigram, Y_train, cv=5))
        print(cross_val_score(sgd, X_train_bigram, Y_train, cv=5))
        print(cross_val_score(sgd, X_train_tf_unigram, Y_train, cv=5))
        print(cross_val_score(sgd, X_train_tf_bigram, Y_train, cv=5))
        
        
        #print(clf.cv_results_['mean_train_score'])
        #clf.predict 
        
