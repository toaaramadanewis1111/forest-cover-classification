# scripts/train_model.py
from preprocess import load_data #using load_data() function from preprocess file
from sklearn.model_selection import train_test_split #split data into a train and testing models
from sklearn.ensemble import RandomForestClassifier #using a tree based ml model
from sklearn.metrics import classification_report, confusion_matrix #evaluate preformance
import matplotlib.pyplot as plt #visualizing results
import seaborn as sns
import joblib #saves model

def train_model():
    X, y = load_data()

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)#spilts model randomly into 70 and 30 percentage,make sure its the same split each time,all classses represented equally

    # Train model
    model = RandomForestClassifier(n_estimators=10, random_state=42)#number of trees in the forest the more the better accuracy but slower training,make sure its the same split each time
    model.fit(X_train, y_train)#trains model

    # Evaluate
    y_pred = model.predict(X_test)#predecits lablel using the train model
    print("Classification Report:\n", classification_report(y_test, y_pred))#shows f1 score and accuracy

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual") #creates a matrix where it shows how my model predicts corrrectly vs incorrectly
    plt.show()

    importances = model.feature_importances_
    sorted_idx = importances.argsort()[-10:]
    plt.figure(figsize=(10,5))
    plt.barh(range(len(sorted_idx)), importances[sorted_idx])
    plt.yticks(range(len(sorted_idx)), X.columns[sorted_idx])
    plt.title("Top 10 Important Features")
    plt.show()


    # Save model
    joblib.dump(model, "models/random_forest.pkl")
    print("✅ Model saved successfully!")

if __name__ == "__main__":
    train_model()
