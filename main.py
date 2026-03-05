from flask import Flask, request, jsonify, render_template
import tensorflow as tf
import numpy as np
import pandas as pd

# initialize flask app
app = Flask(__name__)

# load the trained model and medians
model = tf.keras.models.load_model('diabetes_model.keras')
medians = pd.read_pickle('train_medians.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods = ['POST'])
def predict():
    try:
        # get features from the form
        features = [float(x) for x in request.form.values()]
        final_features = np.array(features).reshape(1, -1)

        new_data = pd.DataFrame(
            final_features,
            columns = [f'Feature{i}' for i in range(1, 9)]
        )

        # fix the missing data again
        col_to_fix = [f'Feature{i}' for i in range (2, 7)]
        new_data[col_to_fix] = new_data[col_to_fix].replace(0, np.nan)
        new_data = new_data.fillna(medians)

        prediction = model.predict(new_data)

        # prepare the output
        output = 'Diabetic' if prediction[0][0] > 0.5 else 'Not Diabetic'
    except:
        output = "Invalid input. Please enter numeric values."

    return render_template('index.html', prediction_text = f'Risk : {output}')

if __name__ == '__main__':
    app.run(debug = True)