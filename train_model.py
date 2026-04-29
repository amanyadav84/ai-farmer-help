import pandas as pd
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
import joblib

# load data
df = pd.read_csv("Crop_recommendation.csv")

X = df.drop('label', axis=1)
y = df['label']

# encode labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# train model
model = XGBClassifier()
model.fit(X, y_encoded)

# save model + encoder
joblib.dump(model, "crop_model.pkl")
joblib.dump(le, "label_encoder.pkl")