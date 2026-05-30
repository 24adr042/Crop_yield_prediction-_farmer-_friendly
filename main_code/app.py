from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import os

app = Flask(__name__)
CORS(app)


DATASET_PATH = os.path.join(os.path.dirname(__file__), "dataset.csv")

df = pd.read_csv(DATASET_PATH)


df = df.rename(columns={
    "Rainfall_mm":              "Rainfall",
    "Temperature_Celsius":      "Temperature",
    "Fertilizer_Used":          "Fertilizer",
    "Irrigation_Used":          "Irrigation",
    "Weather_Condition":        "Weather",
    "Yield_tons_per_hectare":   "Yield"
})


encoders = {}
cat_cols  = ["Region", "Soil_Type", "Crop", "Weather"]

for col in cat_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    encoders[col] = le

df["Fertilizer"] = df["Fertilizer"].astype(int)
df["Irrigation"]  = df["Irrigation"].astype(int)

feature_cols = [
    "Region", "Soil_Type", "Crop",
    "Rainfall", "Temperature",
    "Fertilizer", "Irrigation",
    "Weather", "Days_to_Harvest"
]

X = df[feature_cols].values
y = df["Yield"].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = GradientBoostingRegressor(
    n_estimators=300,
    learning_rate=0.08,
    max_depth=5,
    subsample=0.85,
    min_samples_leaf=5,
    random_state=42
)
model.fit(X_train, y_train)


y_pred = model.predict(X_test)

mae  = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2   = r2_score(y_test, y_pred)

print("=" * 50)
print("       MODEL TRAINING COMPLETE")
print("=" * 50)
print(f"  Dataset size   : {len(df)} samples")
print(f"  Train samples  : {len(X_train)}")
print(f"  Test  samples  : {len(X_test)}")
print("-" * 50)
print(f"  MAE            : {mae:.4f}  tonnes/hectare")
print(f"  RMSE           : {rmse:.4f}  tonnes/hectare")
print(f"  R² Score       : {r2:.4f}  ({r2*100:.2f}% accuracy)")
print("=" * 50)



def safe_encode(col: str, value: str):
    """Encode a single value; raise ValueError with a friendly message."""
    le = encoders[col]
    if value not in le.classes_:
        raise ValueError(
            f"Unknown {col} value '{value}'. "
            f"Allowed: {list(le.classes_)}"
        )
    return int(le.transform([value])[0])


@app.route("/predict", methods=["POST"])
def predict():
    try:
        body = request.json

        # --- parse & validate inputs ---
        region   = body["region"]
        soil     = body["soil_type"]
        crop     = body["crop"]
        rainfall = float(body["rainfall"])
        temp     = float(body["temperature"])
        fert     = 1 if str(body["fertilizer"]).lower() in ["true", "1", "yes"] else 0
        irrig    = 1 if str(body["irrigation"]).lower()  in ["true", "1", "yes"] else 0
        weather  = body["weather"]
        days     = int(body["days_to_harvest"])

        # --- encode categoricals ---
        region_enc  = safe_encode("Region",    region)
        soil_enc    = safe_encode("Soil_Type", soil)
        crop_enc    = safe_encode("Crop",      crop)
        weather_enc = safe_encode("Weather",   weather)

        features   = np.array([[
            region_enc, soil_enc, crop_enc,
            rainfall, temp, fert, irrig,
            weather_enc, days
        ]])

        prediction = float(model.predict(features)[0])
        prediction = max(0.1, round(prediction, 4))

        return jsonify({
            "yield":  prediction,
            "unit":   "tonnes/hectare",
            "status": "success"
        })

    except KeyError as e:
        return jsonify({"error": f"Missing field: {e}", "status": "error"}), 400
    except Exception as e:
        return jsonify({"error": str(e), "status": "error"}), 400


@app.route("/options", methods=["GET"])
def options():
    """Return all valid dropdown values decoded from the encoders."""
    return jsonify({
        "regions"    : sorted(encoders["Region"].classes_.tolist()),
        "soil_types" : sorted(encoders["Soil_Type"].classes_.tolist()),
        "crops"      : sorted(encoders["Crop"].classes_.tolist()),
        "weathers"   : sorted(encoders["Weather"].classes_.tolist()),
        "fertilizer" : [True, False],
        "irrigation" : [True, False]
    })


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status":  "ok",
        "model":   "GradientBoostingRegressor",
        "samples": len(df),
        "r2":      round(r2, 4)
    })


if __name__ == "__main__":
    app.run(debug=True, port=5000)