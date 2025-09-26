import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, r2_score

def load_and_train():
    exercise_df = pd.read_csv("Exercise.csv")
    calories_df = pd.read_csv("Calories.csv")
    df = pd.merge(exercise_df, calories_df, on='User_ID')

    df['Gender'] = df['Gender'].map({'male': 0, 'female': 1})
    df['BMI'] = df['Weight'] / ((df['Height'] / 100) ** 2)
    X = df[['Gender', 'Age', 'Height', 'Weight', 'Duration', 'Heart_Rate', 'Body_Temp', 'BMI']]
    y = df['Calories']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(n_estimators=100),
        "XGBoost": XGBRegressor(n_estimators=100, learning_rate=0.1)
    }

    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        results[name] = {
            "MAE": mean_absolute_error(y_test, preds),
            "R2": r2_score(y_test, preds),
            "Model": model
        }

    return results 
