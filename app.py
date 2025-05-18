from flask import Flask, render_template, request
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.initializers import Orthogonal
import pickle

app = Flask(__name__)


model = tf.keras.models.load_model('fake_news_lstm_model.h5',
                                   custom_objects={'Orthogonal': Orthogonal})
# Load the tokenizer
with open('tokenizer.pkl', 'rb') as handle:
    tokenizer = pickle.load(handle)

max_len = 200  # Same max length as used in training

def preprocess_text(text):
    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=max_len, padding='post')
    return padded

@app.route('/', methods=['GET', 'POST'])
def home():
    prediction = None
    if request.method == 'POST':
        title = request.form['title'] or ''
        content = request.form['content'] or ''
        full_text = title + " " + content
        processed_text = preprocess_text(full_text)
        pred_prob = model.predict(processed_text)[0][0]
        prediction = "Real" if pred_prob > 0.5 else "Fake"
    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
