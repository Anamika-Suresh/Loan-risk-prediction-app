from flask import Flask, render_template, request
import pandas as pd
import pickle

app = Flask(__name__)

with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaling.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("feature_columns.pkl", "rb") as f:
    feature_columns = pickle.load(f)

with open("encoders.pkl", "rb") as f:
    encoders = pickle.load(f)

loan_meaning = {
    "G": "Good",
    "B": "Bad",
    "SS": "Sub-Standard",
    "DF": "Doubtful"
}


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        input_dict = {col: 0 for col in feature_columns}

        # numeric
        input_dict["INVESTMENT_TOTAL"] = float(request.form["INVESTMENT_TOTAL"])
        input_dict["ACCCURRENTBALANCE"] = float(request.form["ACCCURRENTBALANCE"])
        input_dict["INSTALL_SIZE"] = float(request.form["INSTALL_SIZE"])
        input_dict["DUE_PAYMENT"] = float(request.form["DUE_PAYMENT"])

        # label encoded
        marital_value = request.form["INF_MARITAL_STATUS"]
        gender_value = request.form["INF_GENDER"]
        client_type_value = str(request.form["CLIENT_TYPE"])

        input_dict["INF_MARITAL_STATUS"] = encoders["INF_MARITAL_STATUS"].transform([marital_value])[0]
        input_dict["INF_GENDER"] = encoders["INF_GENDER"].transform([gender_value])[0]
        input_dict["CLIENT_TYPE"] = encoders["CLIENT_TYPE"].transform([client_type_value])[0]

        # binary dummy handling based on columns that actually exist
        compensation_value = request.form["COMPENSATION_CHARGED"]
        repay_value = request.form["REPAY_MODE"]

        for col in feature_columns:
            if col.startswith("COMPENSATION_CHARGED_"):
                suffix = col.replace("COMPENSATION_CHARGED_", "")
                input_dict[col] = 1 if compensation_value == suffix else 0

            if col.startswith("REPAY_MODE_"):
                suffix = col.replace("REPAY_MODE_", "")
                input_dict[col] = 1 if repay_value == suffix else 0

        input_df = pd.DataFrame([input_dict], columns=feature_columns)

        print("\nProcessed input:")
        print(input_df)

        input_scaled = scaler.transform(input_df)

        pred_encoded = model.predict(input_scaled)[0]
        pred_label = encoders["QUALITY_OF_LOAN"].inverse_transform([pred_encoded])[0]

        result = f"Predicted Quality of Loan: {pred_label} - {loan_meaning.get(pred_label, '')}"
        return render_template("index.html", prediction_text=result)

    except Exception as e:
        return render_template("index.html", prediction_text=f"Error: {str(e)}")


if __name__ == "__main__":
    app.run(debug=True)

