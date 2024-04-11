from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

# Load the pre-trained model
model_1 = joblib.load(r'C:\Users\Dell\Desktop\Next Hikes Project_06\DecisionTreeRegressor_model.pkl')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Extract input features from the form
    store = int(request.form['Store'])
    day_of_week = int(request.form['DayOfWeek'])
    customers = int(request.form['Customers'])
    Open = int(request.form['Open'])
    promo = int(request.form['Promo'])
    is_holiday = int(request.form['IsHoliday'])
    school_holiday = int(request.form['SchoolHoliday'])
    weekday = int(request.form['Weekday'])
    is_weekend = int(request.form['IsWeekend'])

    # Make predictions
    prediction = model_1.predict([[store, day_of_week, customers, Open, promo, is_holiday, school_holiday, weekday, is_weekend]])

    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
