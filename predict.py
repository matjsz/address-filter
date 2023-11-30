from train import model

input_text = "Rua Amarildo Romari, número 345 jardim gabriel tenório, jundiaí - são paulo, CEP 13245689"

input_seq = tokenizer.texts_to_sequences([input_text])
input_seq_pad = pad_sequences(input_seq, maxlen=train_seq_n_abrev_pad.shape[1])

predicted_sequence = model.predict([input_seq_pad, input_seq_pad])[0]

predicted_text = ' '.join([tokenizer.index_word.get(idx, '') for idx in np.argmax(predicted_sequence, axis=-1)])

print("Texto de entrada original:", input_text)
print("Texto abreviado predito:", predicted_text)