import zipfile
import pandas as pd
import os
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer


zip_file_path = r"C:\Users\ashsi\Downloads\dataset and all.zip"  
extracted_dir = 'extracted_files'

with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
    zip_ref.extractall(extracted_dir)


files = os.listdir(extracted_dir)
print("Files in extracted directory:", files)  

dataframes = {}
for file in files:
    if file.endswith('.csv'):
        file_path = os.path.join(extracted_dir, file)
        df_name = os.path.splitext(file)[0]
        try:
            dataframes[df_name] = pd.read_csv(file_path)
            print(f"Loaded {df_name} successfully.")  
        except Exception as e:
            print(f"Failed to load {file}: {e}")  


print("Loaded dataframes:", dataframes.keys())

required_keys = ['training_set_features', 'training_set_labels', 'test_set_features', 'submission_format']
for key in required_keys:
    if key not in dataframes:
        raise KeyError(f"Key '{key}' not found in loaded dataframes.")


train_features = dataframes['training_set_features']
train_labels = dataframes['training_set_labels']
test_features = dataframes['test_set_features']
submission_format = dataframes['submission_format']


train_data = train_features.merge(train_labels, on='respondent_id')


numerical_features = ['xyz_concern', 'xyz_knowledge', 'opinion_xyz_vacc_effective', 
                      'opinion_xyz_risk', 'opinion_xyz_sick_from_vacc', 
                      'opinion_seas_vacc_effective', 'opinion_seas_risk', 
                      'opinion_seas_sick_from_vacc']
categorical_features = ['age_group', 'education', 'race', 'sex', 'income_poverty', 
                        'marital_status', 'rent_or_own', 'employment_status', 
                        'hhs_geo_region', 'census_msa', 'employment_industry', 
                        'employment_occupation']


preprocessor = ColumnTransformer(
    transformers=[
        ('num', SimpleImputer(strategy='median'), numerical_features),
        ('cat', Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ]), categorical_features)
    ])


X = train_data.drop(columns=['respondent_id', 'xyz_vaccine', 'seasonal_vaccine'])
y = train_data[['xyz_vaccine', 'seasonal_vaccine']]


X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)


model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', MultiOutputClassifier(RandomForestClassifier(n_estimators=100, random_state=42)))
])


print("Training the model...")
model.fit(X_train, y_train)


y_val_pred = model.predict_proba(X_val)


y_val_pred_prob = np.vstack([y[:, 1] for y in y_val_pred]).T

roc_auc = roc_auc_score(y_val, y_val_pred_prob, average='macro')
print(f'Validation ROC AUC Score: {roc_auc:.4f}')


X_test = test_features.drop(columns=['respondent_id'])
test_preds = model.predict_proba(X_test)


submission = pd.DataFrame({
    'respondent_id': test_features['respondent_id'],
    'xyz_vaccine': [pred[1] for pred in test_preds[0]],  
    'seasonal_vaccine': [pred[1] for pred in test_preds[1]] 
})


submission.to_csv('submission.csv', index=False)

print("Submission file created successfully.")
