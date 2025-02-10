Technical Description – MacBook Product Description Generator
The MacBook Product Description Generator is a machine learning-powered web application built using Streamlit. It generates professional, structured product descriptions based on user-inputted features, ensuring variability and relevance.

🔹 Key Functionalities
Accepts five product features as input.
Generates a dynamic, well-structured product description.
Uses randomized sentence structures for variation.
Displays feature frequency analysis using bar charts.
🔹 Technology Stack
Frontend & UI: Streamlit
Machine Learning: Scikit-learn (Random Forest Regressor)
Text Processing: TF-IDF Vectorization
Data Handling: Pandas, NumPy
Visualization: Matplotlib
🔹 Model Workflow
1️⃣ Dataset Creation

A dataset of MacBook features and corresponding descriptions was manually curated.
Features were combined into a single text representation for training.

2️⃣ Preprocessing & Training

TF-IDF Vectorization was used to transform text features into numerical representations.
Label Encoding was applied to descriptions.
A Random Forest Regressor was trained to predict descriptions based on input features.

3️⃣ Description Generation

User-provided features are vectorized using the trained TF-IDF model.
The Random Forest model predicts the best-matching description.
The output description is randomly structured using predefined sentence templates to add variety.

4️⃣ Data Visualization

A bar chart displays the frequency of selected features in the dataset.
