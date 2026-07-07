from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()
STOPWORDS = set(stopwords.words("english"))
import nltk

nltk.download("punkt")
nltk.download("stopwords")

def transform_text(text):

    text = text.lower()

    text = word_tokenize(text)

    y = []

    for word in text:
        if word.isalnum():
            y.append(word)

    text = y
    y = []

    for word in text:
        if word not in STOPWORDS:
            y.append(word)

    text = y
    y = []

    for word in text:
        y.append(ps.stem(word))

    return " ".join(y)