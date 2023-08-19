import pickle
import streamlit as st

# Load the pickled models
with open('model.pkl', 'rb') as model_file:
    svm_classifier = pickle.load(model_file)

with open('Vectorizer.pkl', 'rb') as vectorizer_file:
    tfidf_vectorizer = pickle.load(vectorizer_file)


# Create a Streamlit interface
st.title("SPAM SMS DETECTION")

# Add user input elements (e.g., text input, button)
user_input = st.text_area("Enter a text for classification:")

if st.button("Predict"):
    # Preprocess the user input
    processed_input = tfidf_vectorizer.transform([user_input])

    # Perform classification using the SVM model
    prediction = svm_classifier.predict(processed_input)

    # Display the prediction
    result = prediction[0]
    st.write("Prediction:", result)

    # Check the prediction and display the result
    if result == 1:
        st.header("Spam")
    else:
        st.header("Not Spam")
