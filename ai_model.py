import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

# Load dataset
data = pd.read_csv("ai_query_deflection.csv")

# Features and labels
X = data["query"]
y = data["complexity"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build model pipeline
model = Pipeline([
 ("vectorizer", CountVectorizer()),
 ("classifier", MultinomialNB())
])

# Train model
model.fit(X_train, y_train)

# Test accuracy
accuracy = model.score(X_test, y_test)
print(f"Model Accuracy: {accuracy:.2f}")

# Test with new query
def predict_query(query):
 prediction = model.predict([query])[0]
 print(f"Query: {query}")
 print(f"Predicted Complexity: {prediction}")

# Example
predict_query("Where is my payment?")
predict_query("Unable to update catalog listing")

# Install Python packages
bash
pip instal pandas scikit-learn

# run file
python ai-model.py
