import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Load the dataset
train_df = pd.read_csv("Titanic_train.csv")

# Preprocess the data
train_df['Age'].fillna(train_df['Age'].median(), inplace=True)
train_df['Sex'] = train_df['Sex'].map({'male': 0, 'female': 1})
train_df['Embarked'].fillna(train_df['Embarked'].mode()[0], inplace=True)
train_df = pd.get_dummies(train_df, columns=['Embarked'], prefix='Embarked')

# Select features and target
features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked_C', 'Embarked_Q', 'Embarked_S']
X = train_df[features]
y = train_df['Survived']

# Split data
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Streamlit app
st.title("Titanic Survival Prediction")

# Input fields
pclass = st.selectbox("Passenger Class", [1, 2, 3]) # where 1 is first class, 2 is second class, and 3 is third class
sex = st.radio("Sex", ["Male", "Female"])
age = st.number_input("Age", min_value=0, max_value=100)
sibsp = st.number_input("Number of Siblings/Spouses", min_value=0)
parch = st.number_input("Number of Parents/Children", min_value=0)
fare = st.number_input("Fare", min_value=0.0)
embarked = st.selectbox("Embarked", ["C", "Q", "S"])

# Convert inputs
sex = 1 if sex == "Female" else 0
embarked_c = 1 if embarked == "C" else 0
embarked_q = 1 if embarked == "Q" else 0
embarked_s = 1 if embarked == "S" else 0

# Create input array
input_data = [[pclass, sex, age, sibsp, parch, fare, embarked_c, embarked_q, embarked_s]]

# Make prediction
prediction = model.predict(input_data)[0]
probability = model.predict_proba(input_data)[0][1]

# Display prediction
if st.button("Predict"):
    if prediction == 1:
        st.success(f"This passenger would have survived with a probability of {probability:.2f}")
    else:
        st.error(f"This passenger would not have survived with a probability of {1-probability:.2f}")