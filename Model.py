#NUMPY AND PANDAS FOR DATA STORAGE
import pandas as pd
import numpy as np
#REGULAR EXPRESSION (re) USED FOR REMOVAL OF PARENTHESIS AND EXCLAMATION MARKS IN DATA
import re
#Natural Language Toolkit (NLTK) used for stopword removal and lemmatization
import nltk
from nltk.corpus import stopwords
stopwords=set(stopwords.words('english'))
from nltk.stem import WordNetLemmatizer
lemm=WordNetLemmatizer()
import matplotlib.pyplot as plt
from sklearn import metrics
#OS used to remove the warning in the terminal
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

#Tensorflow used for model creation and traning
from tensorflow.keras.preprocessing.text import Tokenizer, tokenizer_from_json
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Embedding, LSTM, Dense

#JSON used to store tokenizer values
import json
#CODE BEGINS
#Reading Data in PANDAS DataFrame (Total rows in data (approximate) 4,00,000)
df=pd.read_csv(r"Data\test.csv",usecols=["label","review"],dtype={"label":str,"review":str})
#Following 3:1 division of data for training and testing and converting to NUMPY array for feeding to model
labels=np.array(df.label)
y_train=np.array(labels[0:299999])
reviews=np.array(df.review)
x_train=np.array(reviews[0:299999])
y_test=np.array(labels[300001:399999])
x_test=np.array(reviews[300001:399999])

#Pre-processing Data
def preprocess(sentence):
    #Conerting Sentence To Lowercase
    sentence=sentence.lower()
    sentence=re.sub(pattern=r'[^\w\s]',repl='',string=sentence) #Removal of symbols
    
    #Removing Stopwords
    words=sentence.split()
    words=[word for word in words if word not in stopwords]

    #Lemmatize
    words=[lemm.lemmatize(word) for word in words]

    #Joining Words to form a sentence
    sentence=" ".join(words)

    #returning sentence
    return sentence

x_train=[preprocess(str(data)) for data in x_train]
x_test=[preprocess(str(data)) for data in x_test]

#Label Creation
def labeling(text):
    if text=="__label__2":
        return int(1)
    elif text=="__label__1":
        return int(0)
    else:
        return int(-1)
y_train=np.array([labeling(str(label)) for label in y_train])
y_test=np.array([labeling(str(label)) for label in y_test])

#Creating Tokenizer
tokenizer=Tokenizer()
tokenizer.fit_on_texts(x_train)

#Checking Number of Unique Words
num=len(tokenizer.word_index)+1

#Converting Texts To Sequences based on tokenizer
sequences=tokenizer.texts_to_sequences(x_train)

#Passing Sequences so that they are of equal length
data=pad_sequences(sequences, maxlen=30)
x_train=np.array(data)

#Loading already saved .keras model
model=load_model("my_model1.keras")

#Code for model
"""
model=Sequential()
model.add(Embedding(input_dim=num, output_dim=100))
model.add(LSTM(20))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=1, batch_size=19)
"""

#Evaluating Model
loss, accuracy = model.evaluate(x_train,y_train)
print(f'Loss: {loss}')
print(f'Accuracy: {accuracy}')
#Making Predictions With Model
#Converting Test Data to Sequence
test_sequence= tokenizer.texts_to_sequences(x_test)
test_pad=pad_sequences(test_sequence, maxlen=30)
x_test=np.array(test_pad)
prediction = model.predict(x_test)

#print(prediction[0:10],y_test[0:10])
def predicting(num):
    if num>=0.5:
        return int(1)
    else:
        return int(0)

prediction=[predicting(i) for i in prediction]
predict_neg=0
predict_pos=0
for i in prediction:
    if i==0:
        predict_pos=predict_pos+1
    else:
        predict_neg=predict_neg+1
#print(len(y_test),len(prediction))
correct=0
incorrect=0
for i in range(0,98999):
    if prediction[i]==y_test[i]:
        correct+=1
    else:
        incorrect+=1
print("Correct Predictions:",correct)
print("Incorrect Predictions:",incorrect)
acc=metrics.accuracy_score(y_test,prediction)
print(f"Accuracy:{acc}")
pre=metrics.precision_score(y_test,prediction)
print(f"Prediction:{pre}")
sens=metrics.recall_score(y_test,prediction)
print("Sensitivity (How well model predicts positive):",sens )

"""text=["The product is great, had some issue but is good overall"]
test_sequence= tokenizer.texts_to_sequences(text)
test_pad=pad_sequences(test_sequence, maxlen=30)
prediction = model.predict(test_pad)
print(prediction[0])"""

confusion=metrics.confusion_matrix(y_test,prediction)
cm_display=metrics.ConfusionMatrixDisplay(confusion_matrix=confusion,display_labels=["Negative","Positive"])
cm_display.plot()
plt.show()

"""tokenizer_json = tokenizer.to_json()
with open('tokenizer.json', 'w', encoding='utf-8') as f:
    f.write(json.dumps(tokenizer_json, ensure_ascii=False))"""