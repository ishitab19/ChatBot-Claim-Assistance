import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import pickle

def train_model():
    data = pd.read_csv("claim_history.csv")

    X = data[["claim_amount", "policy_years", "hospital_network"]]
    y = data["approved"]

    model = RandomForestClassifier()
    model.fit(X, y)

    with open("claim_model.pkl", "wb") as f:
        pickle.dump(model, f)

    print("Model trained and saved successfully!")

def predict_claim(amount, years, network):
    with open("claim_model.pkl", "rb") as f:
        model = pickle.load(f)

    prediction = model.predict([[amount, years, network]])
    return "Likely Approved" if prediction[0] == 1 else "Risk of Rejection"


# 👇 THIS PART IS VERY IMPORTANT
if __name__ == "__main__":
    train_model()
