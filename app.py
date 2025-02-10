import streamlit as st
import pandas as pd
import numpy as np
import random
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("macbook_features_dataset.csv")
df["Features"] = df[["Feature_1", "Feature_2", "Feature_3", "Feature_4", "Feature_5"]].apply(lambda x: " ".join(x), axis=1)

# Encode descriptions
label_encoder = LabelEncoder()
df["Description_Encoded"] = label_encoder.fit_transform(df["Description"])

# Vectorize text
tfidf = TfidfVectorizer()
X = tfidf.fit_transform(df["Features"])
y = df["Description_Encoded"]

# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

# Streamlit UI
st.title("MacBook Product Description Generator")
st.write("Enter five features to generate a detailed, professional product description.")

# User input
feature_inputs = [st.text_input(f"Feature {i+1}") for i in range(5)]
if st.button("Generate Description"):
    input_text = " ".join(feature_inputs)
    input_vector = tfidf.transform([input_text])
    pred_encoded = int(model.predict(input_vector)[0])
    description = label_encoder.inverse_transform([pred_encoded])[0]
    
    # Randomized sentence structures
    templates = [
        f"Experience the power of {feature_inputs[0]}, combined with {feature_inputs[1]} for seamless performance. "
        f"Enjoy lightning-fast speed with {feature_inputs[2]}, while {feature_inputs[3]} delivers an outstanding visual experience. "
        f"Powered by {feature_inputs[4]}, this MacBook is designed for professionals who demand the best.",
        
        f"With {feature_inputs[0]} at its core, this MacBook redefines speed and efficiency. "
        f"Multitasking is effortless with {feature_inputs[1]}, and {feature_inputs[2]} ensures ample storage. "
        f"The {feature_inputs[3]} brings images to life, making this the ultimate device powered by {feature_inputs[4]}.",
        
        f"Take your workflow to the next level with {feature_inputs[0]}, backed by {feature_inputs[1]} for uninterrupted performance. "
        f"The {feature_inputs[2]} provides ultra-fast storage, while {feature_inputs[3]} makes visuals stunning. "
        f"Running on {feature_inputs[4]}, this MacBook is a powerhouse of innovation."
    ]
    
    long_description = random.choice(templates)
    
    st.subheader("Generated Professional Product Description:")
    st.write(long_description)

    # Visualization
    feature_counts = {feature: df["Features"].str.contains(feature).sum() for feature in feature_inputs}
    plt.figure(figsize=(8, 4))
    plt.bar(feature_counts.keys(), feature_counts.values(), color='skyblue')
    plt.xlabel("Features")
    plt.ylabel("Occurrence in Dataset")
    plt.title("Feature Occurrences in Dataset")
    st.pyplot(plt)
