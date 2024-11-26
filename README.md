# codesoftML
#task-1
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
data = pd.read_csv('movies.csv')  # Example dataset
X = data['plot']  # Plot summaries
y = data['genre']  # Movie genres (multiclass labels)
vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
X_tfidf = vectorizer.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.2, random_state=42)
naive_bayes = MultinomialNB()
naive_bayes.fit(X_train, y_train)
log_reg = LogisticRegression(max_iter=200)
log_reg.fit(X_train, y_train)
svm = SVC(kernel='linear')
svm.fit(X_train, y_train)
y_pred = log_reg.predict(X_test)  # Change to naive_bayes.predict(X_test) or svm.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
