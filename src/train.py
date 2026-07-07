import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from src.config import (MODEL_PATH,DATA_PATH,TFIDF_PATH, TEST_SIZE, RANDOM_STATE, MAX_FEATURES,CV_FOLDS) 
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

from src.preprocessing import transform_text
from src.utils import save_pickle

df = pd.read_csv(DATA_PATH, encoding="latin-1")
df = df.drop(columns=["Unnamed: 2", "Unnamed: 3", "Unnamed: 4"])
df = df.rename(columns={
    "v1": "label",
    "v2": "message"
})
df = df.drop_duplicates()

df["label"] = df["label"].map({
    "ham": 0,
    "spam": 1
})
df["transformed_message"] = df["message"].apply(transform_text)
X = df["transformed_message"]
y = df["label"]
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=TEST_SIZE,random_state=RANDOM_STATE)
tfidf = TfidfVectorizer(max_features=MAX_FEATURES)
X_train = tfidf.fit_transform(X_train)
X_test = tfidf.transform(X_test)
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB

models = {
    "Logistic Regression": LogisticRegression(),
    "Multinomial Naive Bayes": MultinomialNB()
}
best_model = None
best_precision = 0
best_model_name = ""

for name, model in models.items():
    print("=" * 60)
    print(name)
    print("=" * 60)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    cv = cross_val_score(model,X_train,y_train,cv = CV_FOLDS,scoring = "accuracy")
    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")
    print(f"Cross Validation Accuracy : {cv.mean():.4f}")
    print()

    print(classification_report(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))
    print()

    if precision > best_precision:
        best_precision = precision
        best_model = model
        best_model_name = name

save_pickle(
    tfidf,TFIDF_PATH
)
save_pickle(
    best_model,MODEL_PATH
)
print("=" * 60)
print(f"Best Model : {best_model_name}")
print(f"Best Precision : {best_precision:.4f}")
print("=" * 60)