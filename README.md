# codesoftML
#task-1
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

# Load the dataset (this assumes the dataset has columns 'plot' and 'genre')
# Replace this with your own dataset loading method
data = pd.read_csv('movies.csv')  # Example dataset
X = data['plot']  # Plot summaries
y = data['genre']  # Movie genres (multiclass labels)

# Step 1: Preprocessing (optional, depending on dataset)
# For simplicity, assume the text is pre-cleaned and tokenized.

# Step 2: Convert text to numerical features using TF-IDF
vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
X_tfidf = vectorizer.fit_transform(X)

# Step 3: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.2, random_state=42)

# Step 4: Choose a classifier and train the model

# Option 1: Naive Bayes
naive_bayes = MultinomialNB()
naive_bayes.fit(X_train, y_train)

# Option 2: Logistic Regression (use either Naive Bayes or Logistic Regression)
log_reg = LogisticRegression(max_iter=200)
log_reg.fit(X_train, y_train)

# Option 3: Support Vector Machines (SVM)
svm = SVC(kernel='linear')
svm.fit(X_train, y_train)

# Step 5: Evaluate the model
# You can choose one model and evaluate it here
y_pred = log_reg.predict(X_test)  # Change to naive_bayes.predict(X_test) or svm.predict(X_test)

# Calculate accuracy and other evaluation metrics
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
