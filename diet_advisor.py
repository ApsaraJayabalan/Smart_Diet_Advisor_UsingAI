# diet_advisor.py

# Function to gather user information via console input
def get_user_info():
    user_info = {}
    # Prompt the user for their name
    user_info['name'] = input("Enter your name: ")
    # Prompt the user for their age and convert to integer
    user_info['age'] = int(input("Enter your age: "))
    # Prompt the user for their gender
    user_info['gender'] = input("Enter your gender (M/F): ")
    # Prompt the user for any health conditions, split by comma into a list
    user_info['health_conditions'] = input(
        "Enter any health conditions (comma-separated, e.g., diabetes, hypertension): ").split(',')
    # Prompt the user for dietary preferences, split by comma into a list
    user_info['dietary_preferences'] = input("Enter any dietary preferences (e.g., vegetarian, vegan): ").split(',')
    return user_info


# Function to provide dietary recommendations based on user information
def provide_dietary_recommendations(user_info):
    recommendations = []

    # Check for specific health conditions and append recommendations
    if 'diabetes' in user_info['health_conditions']:
        recommendations.append("Limit carbohydrate intake and avoid sugary foods.")
    if 'hypertension' in user_info['health_conditions']:
        recommendations.append("Reduce salt intake and eat more potassium-rich foods.")
    if 'cardiovascular' in user_info['health_conditions']:
        recommendations.append("Consume heart-healthy fats like omega-3s and avoid trans fats.")
    # Check for dietary preferences and append recommendations
    if 'vegetarian' in user_info['dietary_preferences']:
        recommendations.append("Ensure you get enough protein from plant-based sources like beans and lentils.")
    if 'vegan' in user_info['dietary_preferences']:
        recommendations.append("Consider B12 supplementation and include fortified foods in your diet.")
    return recommendations


# Function to generate a meal plan based on dietary recommendations
def generate_meal_plan(recommendations):
    meal_plan = {}

    # Generate meal plan based on specific recommendations
    if any("Limit carbohydrate" in rec for rec in recommendations):
        meal_plan['breakfast'] = "Scrambled eggs with avocado and a side of berries"
        meal_plan['lunch'] = "Grilled chicken salad with leafy greens and olive oil"
        meal_plan['dinner'] = "Baked salmon with steamed broccoli and quinoa"

    elif any("Reduce salt" in rec for rec in recommendations):
        meal_plan['breakfast'] = "Oatmeal with fresh fruit and a handful of nuts"
        meal_plan['lunch'] = "Grilled tofu stir-fry with mixed vegetables"
        meal_plan['dinner'] = "Lentil soup with a side of steamed spinach"

    else:
        meal_plan['breakfast'] = "Smoothie with spinach, banana, and almond butter"
        meal_plan['lunch'] = "Quinoa salad with chickpeas, cucumber, and feta"
        meal_plan['dinner'] = "Grilled veggies with tofu and brown rice"

    return meal_plan


# Main function to run the console application
def main():
    user_info = get_user_info()  # Gather user information
    recommendations = provide_dietary_recommendations(user_info)  # Get dietary recommendations
    meal_plan = generate_meal_plan(recommendations)  # Generate meal plan

    # Print personalized dietary recommendations
    print("\nPersonalized Dietary Recommendations:")
    for rec in recommendations:
        print(f"- {rec}")

    # Print the sample meal plan
    print("\nSample Meal Plan:")
    for meal, menu in meal_plan.items():
        print(f"{meal.capitalize()}: {menu}")


# Entry point for the script when executed directly
if __name__ == "__main__":
    main()


# Import necessary libraries for machine learning
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder

# Load the dataset (assumes 'diet_dataset.csv' exists)
data = pd.read_csv('diet_dataset.csv', encoding='ISO-8859-1')

# Preprocess the data: Encode categorical features into numerical format
le = LabelEncoder()
data['gender'] = le.fit_transform(data['gender'])  # Encode gender
data['health_conditions'] = le.fit_transform(data['health_conditions'])  # Encode health conditions
data['dietary_preferences'] = le.fit_transform(data['dietary_preferences'])  # Encode dietary preferences

# Define features (X) and target variable (y)
X = data[['age', 'gender', 'health_conditions', 'dietary_preferences']]
y = data['dietary_recommendations']

# Split data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Decision Tree Classifier on the training data
clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)


# Function to get dietary recommendations based on machine learning model
def get_ml_recommendations(user_info):
    # Create a DataFrame for the user info
    user_df = pd.DataFrame([user_info])
    # Transform user info using the same label encoder
    user_df['gender'] = le.transform(user_df['gender'])
    user_df['health_conditions'] = le.transform(user_df['health_conditions'])
    user_df['dietary_preferences'] = le.transform(user_df['dietary_preferences'])

    # Make a prediction using the trained model
    prediction = clf.predict(user_df)
    return prediction[0]  # Return the predicted recommendation


# Main function for the machine learning portion of the application
def main_ml():
    user_info = get_user_info()  # Gather user information
    ml_recommendation = get_ml_recommendations(user_info)  # Get ML-based dietary recommendation

    # Print the machine learning-based recommendation
    print(f"\nMachine Learning-based Dietary Recommendation: {ml_recommendation}")


# Entry point for the ML portion of the script when executed directly
if __name__ == "__main__":
    main_ml()
