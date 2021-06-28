import re 
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix,accuracy_score
import confusion_plot as cp
import matplotlib.pyplot as plt
import numpy as np
import itertools
stemmer=PorterStemmer()
lemmatizer=WordNetLemmatizer()
corpus=[]

#DATA READ
import pandas as pd
messages=pd.read_csv('./data/SMSSpamCollection',sep="\t",names=['type','message'])
messages.head()

#BAG OF WORDS

#STEP1 Cleaning the text (remove puncuation, lower, stemmatization,lemmatization)
for i in range(len(messages)):
  clean_mes=re.sub('[^a-zA-Z]',' ',messages['message'][i]) 
  clean_mes=clean_mes.lower()
  clean_mes=clean_mes.split()
  clean_mes=[lemmatizer.lemmatize(word) for word in clean_mes if word not in set (stopwords.words('english'))]
  #clean_mes=[stemmer.stem(word) for word in clean_mes if word not in set (stopwords.words('english'))]
  clean_mes=' '.join(clean_mes)
  corpus.append(clean_mes)

#STEP2 create the bag of words (BOW) using CountVectorizer
#cvec=CountVectorizer()
#X=cvec.fit_transform(corpus).toarray()

tfvec=TfidfVectorizer(max_features=5500)
X=tfvec.fit_transform(corpus).toarray()
#print(X.shape)
y=pd.get_dummies(messages['type'])
y=y.iloc[:,1].values #y=1-> normal y=0-> spam

#SPLIT DATASET
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)

#FIT_MODEL
spam_detector=MultinomialNB().fit(X_train,y_train)
#predict the message type
y_pred=spam_detector.predict(X_test)

#ACCURACY & CONFUSION MATRIX
accuracy=accuracy_score(y_pred,y_test)
print(accuracy)

con_mat=confusion_matrix(y_pred,y_test)
np.set_printoptions(precision=2)
# Plot non-normalized confusion matrix
plt.figure()
cp.plot_confusion_matrix(con_mat, classes=['Normal','Spam'],normalize=False,
                      title='Message Classification')
plt.show()
# Plot normalized confusion matrix
'''plt.figure()
plot_confusion_matrix(con_mat, classes=['Normal','Spam'],normalize=True,
                      title='Message Classification')'''
