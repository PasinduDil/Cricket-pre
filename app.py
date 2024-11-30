from flask import Flask, request, jsonify
import pickle
import pandas as pd

# Step 1: Load the Model and Encoder
with open("fitness_score_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("venue_encoder.pkl", "rb") as f:
    encoder = pickle.load(f)

# Step 2: Initialize Flask App
app = Flask(__name__)

# Load original data to get team information
data = pd.read_excel("Modelnew.xlsx")
data = data[['Team', 'TeamID', 'Fitness_Score', 'Venue']]

# Step 3: Define Endpoint for Prediction
@app.route('/predict', methods=['POST'])
def predict_fitness_score():
    try:
        input_data = request.json
        print("Received data:", input_data)  # Log the received data
    except Exception as e:
        return jsonify({"error": f"Failed to parse JSON: {str(e)}"}), 400

    venue = input_data.get('venue')
    if not venue:
        return jsonify({"error": "Please provide a venue"}), 400

    # Encode the venue
    try:
        venue_encoded = encoder.transform([venue])[0]
    except ValueError:
        return jsonify({"error": "Invalid venue provided"}), 400

    # Predict Fitness Score
    prediction = model.predict([[venue_encoded]])
    predicted_score = prediction[0]

    # Find the team with the highest score for this venue
    highest_score_team = data[data['Fitness_Score'] == data['Fitness_Score'].max()]

    # Return the prediction
    return jsonify({
        "predicted_fitness_score": predicted_score,
        "highest_score_team": highest_score_team['Team'].values[0]
    })


# Step 4: Run the Flask App
if __name__ == '__main__':
    app.run(debug=True)