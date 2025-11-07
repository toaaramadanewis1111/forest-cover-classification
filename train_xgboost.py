import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier, plot_importance
import matplotlib.pyplot as plt
from preprocess import load_data


def train_model():
    # -------------------- Load and prepare data --------------------
    X, y = load_data()  # Load features and target labels from preprocess.py

    # Remove rows where target is missing
    mask = y.notna()
    X = X[mask]
    y = y[mask] - 1  # XGBoost uses zero-based class labels

    # Split into training (80%) and testing (20%)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # -------------------- Define XGBoost model --------------------
    model = XGBClassifier(
        n_estimators=200,           # Number of boosting trees
        learning_rate=0.1,          # Step size shrinkage
        max_depth=8,                # Maximum tree depth
        subsample=0.8,              # Row sampling
        colsample_bytree=0.8,       # Column sampling
        random_state=42,            # Reproducibility
        eval_metric='mlogloss'      # Avoids warnings
    )

    # -------------------- Train model --------------------
    print("🚀 Training XGBoost model...")
    model.fit(X_train, y_train)

    # -------------------- Evaluate model --------------------
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print(f"\n✅ XGBoost Accuracy: {acc:.4f}")
    print("\n📊 Classification Report:\n", classification_report(y_test, y_pred))

    # -------------------- Visualize feature importance --------------------
    plt.figure(figsize=(12, 8))
    plot_importance(
        model,
        max_num_features=10,
        importance_type='gain',
        title='Top 10 Important Features'
    )
    plt.show()

    # -------------------- Save model --------------------
    model.save_model("xgboost_model.json")
    print("\n💾 Model saved as 'xgboost_model.json'")

    return model


# -------------------- Run when executed directly --------------------
if __name__ == "__main__":
    train_model()
