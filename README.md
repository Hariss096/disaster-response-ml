# Disaster Response Pipeline Project

### Instructions:
1. Install dependencies:
    - pandas
    - sqlalchemy
    - nltk
    - sklearn
    - plotly
    - flask
    
2. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

3. Run the following command in the app's directory to run your web app.
    `python run.py`

4. Go to http://0.0.0.0:3001/

# Extras

## Example message classification:

![alt text](https://github.com/Hariss096/disaster-response-ml/blob/master/resources/Example%20message%20with%20highlighted%20categories.png)

## Insights:

![alt text](https://github.com/Hariss096/disaster-response-ml/blob/master/resources/Grouped%20Bar%20Chart.png)


![alt text](https://github.com/Hariss096/disaster-response-ml/blob/master/resources/Distribution%20of%20genres.png)


![alt text](https://github.com/Hariss096/disaster-response-ml/blob/master/resources/most%20frequent%20words.png)
