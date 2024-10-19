# Import necessary modules from Flask framework
from flask import Flask, render_template, request

# Import functions for dietary recommendations and meal plan generation
from diet_advisor import provide_dietary_recommendations, generate_meal_plan
# Create an instance of the Flask class
app = Flask(__name__)
# Define the route for the homepage
@app.route('/')
def index():
    # Render the index1.html template when the homepage is accessed
    return render_template('index1.html')

# Define the route to handle dietary recommendations
@app.route('/get_recommendations', methods=['POST'])
def get_recommendations():
    # Gather user information from the form submission
    user_info = {
        'name': request.form['name'],  # User's name
        'age': int(request.form['age']),  # User's age, converted to integer
        'gender': request.form['gender'],  # User's gender
        'health_conditions': request.form.getlist('health_conditions'),  # List of health conditions
        'dietary_preferences': request.form.getlist('dietary_preferences')  # List of dietary preferences
    }

    # Call the function to provide dietary recommendations based on user info
    recommendations = provide_dietary_recommendations(user_info)

    # Generate a meal plan based on the recommendations
    meal_plan = generate_meal_plan(recommendations)

    # Render the results.html template with recommendations and meal plan data
    return render_template('results.html', recommendations=recommendations, meal_plan=meal_plan)


# Run the Flask application in debug mode if this script is executed directly
if __name__ == '__main__':
    app.run(debug=True)
