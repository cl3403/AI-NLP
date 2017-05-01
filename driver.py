import re
import string
import os
import pandas as pd 


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

                content_list = content.split()
                ' '.join(i for i in content_list if i not in stopwords)
                
                rows.append(row)
                contents.append(content)
                polarity.append(1)
                row += 1

        for filename in os.listdir(neg_path):
                filepath = neg_path+filename
                inF = open(filepath, "r")
                content = inF.read().lower()
                inF.close()
                content = regex1.sub(' ',content)
                content = regex2.sub(' ',content)

                content_list = content.split()
                ' '.join(i for i in content_list if i not in stopwords)
                
                rows.append(row)
                contents.append(content)
                polarity.append(0)
                row += 1

        dataSet = list(zip(rows,contents,polarity))
        df = pd.DataFrame(data = dataSet, columns = ['row_number','text','polarity'])
        df.to_csv(outpath+name, index=False, header=True)
  
if __name__ == "__main__":
        imdb_data_preprocess(train_path)
