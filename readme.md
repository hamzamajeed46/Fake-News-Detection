# Fake News Detection with LSTM

This project detects whether a news article is **Fake** or **Real** using an LSTM deep learning model trained on the WELFake dataset (72,000+ labeled news articles). The model uses both the news **title** and **content** as input and is deployed with a simple Flask web interface for real-time predictions.

---

## Requirements

- Python 3.10+
- TensorFlow 2.16.1
- Flask
- scikit-learn
- numpy
- pandas
- matplotlib

Install dependencies with:

```bash
pip install tensorflow==2.16.1 flask scikit-learn numpy pandas matplotlib
```

## Run Flask Application
Run the Flask app to start the web server:
```bash
python app.py
```

Navigate to:
    http://127.0.0.1:5000/
