from src.preprocessing import transform_text
from src.utils import load_pickle

# Load once
model = load_pickle("models/model.pkl")
tfidf = load_pickle("models/tfidf_vectorizer.pkl")


def predict_message(message):

    # Preprocess
    transformed_message = transform_text(message)

    # Vectorize
    vector = tfidf.transform([transformed_message])

    # Predict
    prediction = model.predict(vector)[0]

    if prediction == 1:
        return "Spam"

    return "Ham"