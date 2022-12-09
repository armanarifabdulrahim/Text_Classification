#Develop the model to classify the text

#%%
# Import module
import os,re,datetime,json,pickle
import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras import Sequential
from tensorflow.keras.layers import LSTM, Input, Dense, Bidirectional, Embedding
from tensorflow.keras.callbacks import TensorBoard
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split



#%%
# Step 1) Data Loading
path = os.path.join(os.getcwd(),'dataset')
df = pd.read_csv(os.path.join(path,'True.csv'))



#%%
# Step 2) Data Inspection
print(df.info())
print(df.describe())
print(df.duplicated().sum())
print(df.isna().sum())
df.head()



#%%
# Step 3) Data Cleaning


for index,i in enumerate(df['text']):
    df['text'][index] = re.sub('(^[^-]*)|(@[^\s]+)|bit.ly/\d.{1,6}|(\s+EST)|([^a-zA-Z])', ' ', i).lower()
    



#%%
# Step 4) Feature Selection


text = df['text']
target = df['subject']

df1 = pd.concat([text, target], axis=1)
df1 = df1.drop_duplicates()

text = df1['text']
target = df1['subject']



#%%
# Step 5) Data Pre-processing
#Tokenizer
#num_words = 500 #Check the median and mean of number of words
words= 5000
tokenizer = Tokenizer(num_words=words,oov_token='<OOV>')
tokenizer.fit_on_texts(text)

word_index = tokenizer.word_index
train_text = tokenizer.texts_to_sequences(text)

#Padding
train_text = pad_sequences(train_text, maxlen=400, padding='post', truncating='post')

#target preprocessing
ohe = OneHotEncoder(sparse=False)
train_subject = ohe.fit_transform(target[::, None])

#train test split
train_text = np.expand_dims(train_text,-1)
X_train, X_test, y_train, y_test = train_test_split(train_text,train_subject)



#%%
# Step 6) Model Development

model = Sequential()
model.add(Embedding(words,128))
model.add(LSTM(128,return_sequences=True))
model.add(LSTM(128))
model.add(Dense(y_train.shape[1],activation='softmax'))
model.summary

model.compile(optimizer='adam',loss='categorical_crossentropy',metrics='accuracy')

logdir = os.path.join(os.getcwd(), 'logs', datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
tb = TensorBoard(log_dir=logdir)

history = model.fit(X_train,y_train, validation_data=(X_test,y_test), epochs=10, callbacks=tb)



#%%
# Step 7) Model Evaluation
y_pred = model.predict(X_test)
y_pred = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_test, axis=1)

print(confusion_matrix(y_true,y_pred))
print(classification_report(y_true,y_pred))
print(accuracy_score(y_true,y_pred))


#%%
# Step 8) Model Saving

with open('token.json','w') as f:
    json.dump(tokenizer.to_json(),f)

with open('onehot.pkl','wb') as f:
    pickle.dump(ohe,f)

model.save('text_classification3.h5')