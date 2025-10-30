from flask import Flask, request, render_template
import numpy as np
import pandas as pd
import logging
import os
from src.Pipeline.predict_pipeline import CustomData, predict_pipeline

# Create logs directory if it doesn't exist
os.makedirs('logs', exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/app.log'),
        logging.StreamHandler()
    ]
)

# Create logger instance
logger = logging.getLogger(__name__)

application = Flask(__name__)
app = application

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('home.html')
    else:
        try:
            # Get form data
            reading_score = request.form.get('reading_score')
            writing_score = request.form.get('writing_score')
            
            if not reading_score or not writing_score:
                logger.warning("Missing required scores")
                return render_template('home.html', error="Please provide both reading and writing scores")
            
            data = CustomData(
                gender=request.form.get('gender'),
                race_ethnicity=request.form.get('ethnicity_group'),
                parental_level_of_education=request.form.get('parental_level_of_education'),
                lunch=request.form.get('lunch'),
                test_preparation_course=request.form.get('test_preparation_course'),    
                reading_score=float(reading_score),
                writing_score=float(writing_score)
            )

            pred_df = data.get_data_as_data_frame()
            logger.info(f"Input DataFrame: \n{pred_df}")

            predict_pipeline_obj = predict_pipeline()
            results = predict_pipeline_obj.predict(pred_df)
            
            prediction = round(float(results[0]), 2)
            logger.info(f"Prediction: {prediction}")
            
            return render_template('home.html', 
                                 prediction=prediction,
                                 reading_score=reading_score,
                                 writing_score=writing_score)
            
        except ValueError as e:
            logger.error(f"Value error: {str(e)}")
            return render_template('home.html', error="Please enter valid numeric scores")
        except Exception as e:
            logger.error(f"Prediction error: {str(e)}")
            return render_template('home.html', error=f"An error occurred: {str(e)}")

if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)