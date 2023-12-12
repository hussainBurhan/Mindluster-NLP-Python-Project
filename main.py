import pandas as pd
import string
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from matplotlib import pyplot

# Set option to display longer text in pandas dataframe
pd.set_option('display.max_colwidth', 100)

# Read data from 'SMSSpamCollection' file and set column names
data = pd.read_csv('SMSSpamCollection', sep='\t', header=None)
data.columns = ['label', 'message']

# Function to remove punctuation from a text
def remove_punct(txt):
    txt_no_punct = ''.join([c for c in txt if c not in string.punctuation])
    return txt_no_punct

# Apply the remove_punct function to the 'message' column and create a new column 'clean_msg'
data['clean_msg'] = data['message'].apply(lambda x: remove_punct(x))

# Function to tokenize a text
def tokenize(txt):
    tokens = re.split('\W+', txt)
    return tokens

# Apply the tokenize function to the 'clean_msg' column and create a new column 'clean_msg_tokenized'
data['clean_msg_tokenized'] = data['clean_msg'].apply(lambda x: tokenize(x.lower()))

# Download stopwords from nltk
nltk.download('stopwords')
stopwords = nltk.corpus.stopwords.words('english')

# Function to remove stopwords from a list of tokens
def remove_stopwords(txt_tokenize):
    txt_clean = [word for word in txt_tokenize if word not in stopwords]
    return txt_clean

# Apply the remove_stopwords function to the 'clean_msg_tokenized' column and create a new column 'no_sw_msg'
data['no_sw_msg'] = data['clean_msg_tokenized'].apply(lambda x: remove_stopwords(x))

# Initialize Porter Stemmer
ps = nltk.PorterStemmer()

# Function to perform stemming on a list of tokens
def stemming(tokenized_txt):
    stemmed_txt = [ps.stem(word) for word in tokenized_txt]
    return stemmed_txt

# Apply the stemming function to the 'no_sw_msg' column and create a new column 'stemmed_msg'
data['stemmed_msg'] = data['no_sw_msg'].apply(lambda x: stemming(x))

# Download WordNet Lemmatizer from nltk
nltk.download('wordnet')
wn = nltk.WordNetLemmatizer()

# Function to perform lemmatization on a list of tokens
def lemmatizer(tokenized_txt):
    lem_txt = [wn.lemmatize(word) for word in tokenized_txt]
    return lem_txt

# Apply the lemmatizer function to the 'no_sw_msg' column and create a new column 'lemmatized_msg'
data['lemmatized_msg'] = data['no_sw_msg'].apply(lambda x: lemmatizer(x))

# Initialize CountVectorizer with lemmatizer as the analyzer
cv = CountVectorizer(analyzer=lemmatizer)
x = cv.fit_transform(data['lemmatized_msg'])
df = pd.DataFrame(x.toarray(), columns=cv.get_feature_names_out())

# Function to clean text by removing punctuation, splitting into tokens, removing stopwords, and stemming
def clean_text(txt):
    txt = ''.join([c for c in txt if c not in string.punctuation])
    tokens = re.split('\W+', txt)
    txt = ' '.join([ps.stem(word) for word in tokens if word not in stopwords])
    return txt

# Apply the clean_text function to the 'message' column and create a new column 'msg_clean'
data['msg_clean'] = data['message'].apply(lambda x: clean_text(x))
cv2 = CountVectorizer(ngram_range=(2, 2))
y = cv2.fit_transform(data['msg_clean'])
df2 = pd.DataFrame(y.toarray(), columns=cv2.get_feature_names_out())

tfidf_vector = TfidfVectorizer(analyzer=lemmatizer)
z = tfidf_vector.fit_transform(data['lemmatized_msg'])
df3 = pd.DataFrame(x.toarray(), columns=tfidf_vector.get_feature_names_out())
print('Count Vectorization')
print(df)
print('N-gram Vectorization')
print(df2)
print('Tf-Idf Vectorization')
print(df3)

# Feature Engineering
data2 = pd.read_csv("SMSSpamCollection", sep='\t')
data2.columns = ['label', 'msg']
print(data2.head())
data2['msg_len'] = data2['msg'].apply(lambda x: len(x))
print(data2.head())

# Function to count the percentage of punctuation in a text
def punctuation_count(txt):
    count = sum([1 for c in txt if c in string.punctuation])
    return 100 * count / len(txt)

# Apply the punctuation_count function to the 'msg' column and create a new column 'punctuation_%'
data2['punctuation_%'] = data2['msg'].apply(lambda x: punctuation_count(x))
print(data2.head())

# Visualization of message length distribution for spam and ham
bins = np.linspace(0, 500, 50)
pyplot.hist(data2[data2['label'] == 'spam']['msg_len'], bins, label='spam', density=True)
pyplot.hist(data2[data2['label'] == 'ham']['msg_len'], bins, label='ham', density=True)
pyplot.legend(loc='upper right')
pyplot.show()

# Visualization of punctuation percentage distribution for spam and ham
bins2 = np.linspace(0, 30, 50)
pyplot.hist(data2[data2['label'] == 'spam']['punctuation_%'], bins2, label='spam', density=True)
pyplot.hist(data2[data2['label'] == 'ham']['punctuation_%'], bins2, label='ham', density=True)
pyplot.legend(loc='upper right')
pyplot.show()  # Not Useful

# Transformation of punctuation percentage
for i in [2, 3, 4]:
    pyplot.hist((data2['punctuation_%']) ** (1 / i), bins=50)
    pyplot.title(f'Transform = 1/{i}')
    pyplot.show()
