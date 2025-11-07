# scripts/preprocess.py
from ucimlrepo import fetch_ucirepo #machine learnining repositrory 
import pandas as pd #to import data as spreedsheets

def load_data():
    # Load Covertype dataset directly from UCI ML Repo
    covertype = fetch_ucirepo(id=31) #can also use covertype=fetch_ucirepo(name="covertype") instead of id
    X = pd.DataFrame(covertype.data.features) #changed to frames
    y = pd.Series(covertype.data.targets.values.ravel(), name="Cover_Type") #Extracts the target labels into a Pandas Series = 1 deimsional array values and their index
    return X, y

if __name__ == "__main__":
    X, y = load_data()
    print("✅ Data loaded successfully!")
    print("Shape:", X.shape, y.shape)
    print("Classes:", y.unique())
