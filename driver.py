import re
import string
import os
import pandas as pd 


train_path = "../aclImdb/train/" # source data
test_path = "../imdb_te.csv" # test data for grade evaluation. 


def imdb_data_preprocess(inpath, outpath="./", name="imdb_tr.csv", mix=False):
 	'''Implement this module to extract
	and combine text files under train_path directory into 
    imdb_tr.csv. Each text file in train_path should be stored 
    as a row in imdb_tr.csv. And imdb_tr.csv should have three 
    columns, "row_number", "text" and label'''

    pos_path = in_path + "pos/"
    neg_path = in_path + "neg/"
    row = 0
    rows = []
    contents = []
    polarity = []

    regex1 = re.compile('<.*?>')
    regex2 = re.compile('[%s0-9]' % re.escape(string.punctuation))

    for filename in os.listdir(pos_path):
            inF = open(filename, "r")
            content = inF.read().lower()
            inF.close()
            content = regex1.sub(' ',content)
            content = regex2.sub(' ',content)
            rows.append(row)
            contents.append(content)
            polarity.append(1)
            row += 1

    for filename in os.listdir(neg_path):
            inF = open(filename, "r")
            content = inF.read().lower()
            inF.close()
            content = regex1.sub(' ',content)
            content = regex2.sub(' ',content)
            rows.append(row)
            contents.append(content)
            polarity.append(0)
            row += 1

    dataSet = list(zip(rows,contents,polarity))
    df = pd.DataFrame(data = dataSet, column = ['row_number','text','polarity']
    df.to_csv(outpath+name, index=False, header=False)
  
if __name__ == "__main__":
 	'''train a SGD classifier using unigram representation,
    predict sentiments on imdb_te.csv, and write output to
    unigram.output.txt'''
  	
    '''train a SGD classifier using bigram representation,
    predict sentiments on imdb_te.csv, and write output to
    unigram.output.txt'''
     
     '''train a SGD classifier using unigram representation
     with tf-idf, predict sentiments on imdb_te.csv, and write 
     output to unigram.output.txt'''
  	
     '''train a SGD classifier using bigram representation
     with tf-idf, predict sentiments on imdb_te.csv, and write 
     output to unigram.output.txt'''
     imdb_data_preprocess（train_path)
