import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense, Attention, Concatenate
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

df = pd.read_csv('dataset.csv')

train_data, test_data = train_test_split(df, test_size=0.2, random_state=42)

tokenizer = Tokenizer(filters='')
tokenizer.fit_on_texts(np.concatenate([train_data['n_abrev'].values, train_data['abrev'].values]))

train_seq_n_abrev = tokenizer.texts_to_sequences(train_data['n_abrev'])
train_seq_abrev = tokenizer.texts_to_sequences(train_data['abrev'])

train_seq_n_abrev_pad = pad_sequences(train_seq_n_abrev)
train_seq_abrev_pad = pad_sequences(train_seq_abrev)

embedding_dim = 50
lstm_units = 100

input_n_abrev = Input(shape=(train_seq_n_abrev_pad.shape[1],))
input_abrev = Input(shape=(train_seq_abrev_pad.shape[1],))

embedding_layer = Embedding(len(tokenizer.index_word) + 1, embedding_dim)

embedding_n_abrev = embedding_layer(input_n_abrev)
embedding_abrev = embedding_layer(input_abrev)

lstm_layer = LSTM(lstm_units, return_sequences=True)

lstm_output_n_abrev = lstm_layer(embedding_n_abrev)
lstm_output_abrev = lstm_layer(embedding_abrev)

attention = Attention()([lstm_output_n_abrev, lstm_output_abrev])

merged_output = Concatenate(axis=-1)([lstm_output_abrev, attention])

output_layer = Dense(len(tokenizer.index_word) + 1, activation='softmax')
output = output_layer(merged_output)

model = Model(inputs=[input_n_abrev, input_abrev], outputs=output)

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit([train_seq_n_abrev_pad, train_seq_abrev_pad], np.array(train_data['abrev']), epochs=10, batch_size=32, validation_split=0.1)

test_seq_n_abrev = tokenizer.texts_to_sequences(test_data['n_abrev'])
test_seq_abrev = tokenizer.texts_to_sequences(test_data['abrev'])

test_seq_n_abrev_pad = pad_sequences(test_seq_n_abrev)
test_seq_abrev_pad = pad_sequences(test_seq_abrev)

loss, accuracy = model.evaluate([test_seq_n_abrev_pad, test_seq_abrev_pad], np.array(test_data['abrev']))
print(f'Test Loss: {loss}, Test Accuracy: {accuracy}')
