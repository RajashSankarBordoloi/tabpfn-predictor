import pandas as pd
import time
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from tabpfn import TabPFNClassifier, TabPFNRegressor


def tabpfn_predict(df: pd.DataFrame, target_column: str, task_type: str):
    try:
        # Drop rows with missing values
        df = df.dropna()

        # Separate features and target
        X = df.drop(columns=[target_column])
        y = df[target_column]

        start_time = time.time()

        # === Classification ===
        if task_type == "Classification":
            label_encoder = LabelEncoder()
            y_encoded = label_encoder.fit_transform(y)

            X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.3, random_state=42)

            clf = TabPFNClassifier()
            clf.fit(X_train, y_train)

            y_pred = clf.predict(X_test)
            y_proba = clf.predict_proba(X_test)
            accuracy = accuracy_score(y_test, y_pred)

            result_df = pd.DataFrame({
                "True": label_encoder.inverse_transform(y_test),
                "Predicted": label_encoder.inverse_transform(y_pred),
                "Confidence": y_proba.max(axis=1)
            })

            return {
                "result_df": result_df,
                "accuracy": accuracy,
                "class_mapping": {k: int(v) for k, v in zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_))},
                "y_proba": y_proba,
                "time_taken": time.time() - start_time
            }

        # === Regression ===
        elif task_type == "Regression":
            X_train, X_test, y_train, y_test = train_test_split(X, y.astype(float), test_size=0.3, random_state=42)

            reg = TabPFNRegressor()
            reg.fit(X_train, y_train)
            y_pred = reg.predict(X_test)

            result_df = pd.DataFrame({
                "True": y_test,
                "Predicted": y_pred
            })

            return {
                "result_df": result_df,
                "mse": mean_squared_error(y_test, y_pred),
                "r2": r2_score(y_test, y_pred),
                "time_taken": time.time() - start_time
            }

        else:
            return None

    except Exception as e:
        print(f"Error: {e}")
        return None
