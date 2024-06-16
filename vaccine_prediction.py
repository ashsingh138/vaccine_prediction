# Install necessary libraries (if not already installed)
# !pip install pandas scikit-learn openpyxl

import zipfile
import pandas as pd
import os
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# Step 1: Unzip the file
zip_file_path = 'C:\Users\ashsi\Downloads\dataset and all.zip'
extracted_dir = 'extracted_files'

with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
    zip_ref.extractall(extracted_dir)

# Step 2: Load the Excel files
files = os.listdir(extracted_dir)
dataframes = {}
for file in files:
    if file.endswith('.xlsx') or file.endswith('.xls'):
        file_path = os.path.join(extracted_dir, file)
        df_name = os.path.splitext(file)[0]
        dataframes[df_name] = pd.read_excel(file_path)

# Display the keys to verify the loaded dataframes
print(dataframes.keys())

# Step 3: Merge training features with labels
train_features = dataframes['training_set_features']
train_labels = dataframes['training_set_labels']
train_data = train_features.merge(train_labels, on='respondent_id')

# Load test features and submission format
test_features = dataframes['test_set_features']
submission_format = dataframes['submission_format']

# Display the first few rows of the combined training data
print(train_data.head())

# Step 4: Preprocessing
# Define feature columns
categorical_features = ['age_group', 'education', 'race', 'sex', 'income_poverty', 
                        'marital_status', 'rent_or_own', 'employment_status', 
                        'hhs_geo_region', 'census_msa', 'employment_industry', 
                        'employment_occupation']

numerical_features = ['xyz_concern', 'xyz_knowledge', 'opinion_xyz_vacc_effective', 
                      'opinion_xyz_risk', 'opinion_xyz_sick_from_vacc', 
                      'opinion_seas_vacc_effective', 'opinion_seas_risk', 
                      'opinion_seas_sick_from_vacc']

# Define the preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        ('num', SimpleImputer(strategy='median'), numerical_features),
        ('cat', Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ]), categorical_features)
    ])

# Define the features and target variables
X_train = train_data.drop(columns=['respondent_id', 'xyz_vaccine', 'seasonal_vaccine'])
y_train = train_data[['xyz_vaccine', 'seasonal_vaccine']]
X_test = test_features.drop(columns=['respondent_id'])

# Create and train the model pipeline
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', MultiOutputClassifier(RandomForestClassifier(n_estimators=100, random_state=42)))
])

model.fit(X_train, y_train)

# Step 5: Make predictions on the test set
y_pred = model.predict_proba(X_test)

# Extract the probabilities
y_pred_proba = pd.DataFrame({
    'xyz_vaccine': y_pred[0][:, 1],
    'seasonal_vaccine': y_pred[1][:, 1]
})

# Prepare the submission file
submission = pd.DataFrame({
    'respondent_id': test_features['respondent_id'],
    'xyz_vaccine': y_pred_proba['xyz_vaccine'],
    'seasonal_vaccine': y_pred_proba['seasonal_vaccine']
})

# Save the submission file
submission.to_csv('submission.csv', index=False)

print("Submission file created successfully.")
