import pandas as pd
import mysql.connector
from sqlalchemy import create_engine, text
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.exceptions import UndefinedMetricWarning
from tabulate import tabulate
import warnings

# Database configuration
DB_HOST = "localhost"
DB_USER = "root"
DB_PASSWORD = "NIKOLATESLA369"
DB_NAME = "school_db"

def create_database():
    # Connect without specifying the database to create it if it doesn't exist
    connection = mysql.connector.connect(
        host=DB_HOST,
        user=DB_USER,
        password=DB_PASSWORD
    )
    cursor = connection.cursor()

    # Create database if it doesn't exist
    cursor.execute("CREATE DATABASE IF NOT EXISTS school_db")
    print("Database 'school_db' created or verified successfully.")
    connection.close()

def create_student_table():
    # Ensure database is created
    create_database()

    # Connect to 'school_db' database
    engine = create_engine(f"mysql+mysqlconnector://{DB_USER}:{DB_PASSWORD}@{DB_HOST}/{DB_NAME}")

    # Define the SQL command to create the students table    
    create_table_query = """
    CREATE TABLE IF NOT EXISTS students (
        Roll_Number INT PRIMARY KEY AUTO_INCREMENT,
        name VARCHAR(100),
        attendance INT,
        hours_studied FLOAT,
        weekly_study_hours FLOAT,
        previous_score FLOAT,
        assignments_completed INT,
        stress_level INT,
        learning_style VARCHAR(20),
        extracurriculars_involved INT,
        goal_score FLOAT,
        score FLOAT
    )
    """

    # Execute the table creation query
    with engine.connect() as connection:
        connection.execute(text(create_table_query))
        print("Table 'students' created or verified successfully.")

def create_db_connection():
    engine = create_engine(f"mysql+mysqlconnector://{DB_USER}:{DB_PASSWORD}@{DB_HOST}/{DB_NAME}")
    return engine

# Display 'students' table data using tabulate
def display_table():
    engine = create_db_connection()
    query = "SELECT * FROM students"
    data = pd.read_sql(query, engine)
    if data.empty:
        print("No data in 'students' table.")
    else:
        print("\nCurrent 'students' Table Data:")
        print(tabulate(data, headers="keys", tablefmt="fancy_grid"))

# Search for a student by roll number
def search_student_by_roll(Roll_Number):
    engine = create_db_connection()
    query = f"SELECT * FROM students WHERE Roll_Number = {Roll_Number}"
    data = pd.read_sql(query, engine)
    if data.empty:
        print(f"No student found with Roll Number: {Roll_Number}")
    else:
        print("\nStudent Details:")
        print(tabulate(data, headers="keys", tablefmt="fancy_grid"))

# Fetch student data from MySQL database for training
def fetch_student_data():
    engine = create_db_connection()
    query = """
        SELECT attendance, hours_studied, weekly_study_hours, previous_score, assignments_completed, 
               stress_level, learning_style, extracurriculars_involved, goal_score, score 
        FROM students
    """
    data = pd.read_sql(query, engine)
    return data

# Insert a student's data into the MySQL database
def insert_student_data(name, attendance, hours_studied, weekly_study_hours, previous_score, assignments_completed, stress_level, learning_style, extracurriculars_involved, goal_score, score):
    engine = create_db_connection()
    try:
        with engine.begin() as connection:
            insert_query = text(""" 
                INSERT INTO students (name, attendance, hours_studied, weekly_study_hours, previous_score, assignments_completed, stress_level, learning_style, extracurriculars_involved, goal_score, score)
                VALUES (:name, :attendance, :hours_studied, :weekly_study_hours, :previous_score, :assignments_completed, :stress_level, :learning_style, :extracurriculars_involved, :goal_score, :score)
            """)

            # Execute the insert statement
            connection.execute(insert_query, {
                "name": name,
                "attendance": attendance,
                "hours_studied": hours_studied,
                "weekly_study_hours": weekly_study_hours,
                "previous_score": previous_score,
                "assignments_completed": assignments_completed,
                "stress_level": stress_level,
                "learning_style": learning_style,
                "extracurriculars_involved": extracurriculars_involved,
                "goal_score": goal_score,
                "score": score
            })
        print("Student data inserted successfully.")
    except Exception as e:
        print(f"Failed to insert data: {e}")

# Function to get data from the user and store it in the database
def get_student_data():
    print("Enter student details:")
    name = input("Name: ")
    attendance = int(input("Attendance (as a percentage): "))
    hours_studied = float(input("Hours Studied: "))
    weekly_study_hours = float(input("Weekly Study Hours: "))
    previous_score = float(input("Previous Score out of 500: "))
    assignments_completed = int(input("Assignments Completed (1-30): "))
    stress_level = int(input("Stress Level (1-10): "))
    learning_style = input("Learning Style (Visual/Auditory/Kinesthetic): ")
    extracurriculars_involved = int(input("Extracurricular Involvement (number of activities): "))
    goal_score = float(input("Goal Score for the term out of 500: "))
    score = float(input("Current Exam Score out of 500: "))
    
    insert_student_data(name, attendance, hours_studied, weekly_study_hours, previous_score, assignments_completed, stress_level, learning_style, extracurriculars_involved, goal_score, score)

# Early Warning System: Identify students at risk of underperforming
def early_warning_system(predicted_score, threshold=50):
    if predicted_score < threshold:
        print("\n⚠️ Warning: Student at risk of underperforming! ⚠️")
        print("📢 Sending notification to parents and teachers...\n")

# Train the model and predict based on user data
def train_and_predict():
    df = fetch_student_data()
    if df.empty:
        print("No data available. Please add data before training the model.")
        return

    # Encode learning styles and drop the original column
    df["learning_style_encoded"] = df["learning_style"].map({"Visual": 1, "Auditory": 2, "Kinesthetic": 3})
    df.drop(columns=["learning_style"], inplace=True)

    # Handle missing values by filling with the median (or you can drop them)
    df.fillna(df.median(numeric_only=True), inplace=True)

    # Define features (X) and target (y)
    X = df.drop(columns=["score"])
    y = df["score"]

    if len(df) < 2:
        print("Not enough data to split. Please add more student data.")
        return

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Predict on test data
    y_pred = model.predict(X_test)

    # Evaluate the model
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UndefinedMetricWarning)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

    print("Model trained successfully.")
    print("Mean Squared Error:", mse)
    print("R-squared Score:", r2 if len(y_test) > 1 else "N/A - Not enough data")

    # Ensure feature names match between training and prediction 
    print("\nEnter details for a new student to predict their score.")
    
    # Create new student data with the correct order of columns
    new_student = pd.DataFrame({
        "attendance": [int(input("Attendance (as a percentage): "))],
        "hours_studied": [float(input("Hours Studied: "))],
        "weekly_study_hours": [float(input("Weekly Study Hours: "))],
        "previous_score": [float(input("Previous Score out of 500: "))],
        "assignments_completed": [int(input("Assignments Completed (1-30): "))],
        "stress_level": [int(input("Stress Level (1-10): "))],
        "learning_style_encoded": [int(input("Learning Style (1=Visual, 2=Auditory, 3=Kinesthetic): "))],
        "extracurriculars_involved": [int(input("Extracurricular Involvement (number of activities): "))],
        "goal_score": [float(input("Goal Score for the term out of 500: "))]
    })

    # Ensure the new_student DataFrame has the same columns as X
    new_student = new_student[X.columns]

    # Fill any missing values
    new_student.fillna(df.median(numeric_only=True), inplace=True)

    # Predict the score for the new student
    predicted_score = model.predict(new_student)[0]
    print(f"Predicted Score: {predicted_score:.2f}")

    # Early warning system
    early_warning_system(predicted_score)

# Main function to run the program
if __name__ == "__main__":
    create_student_table()

    while True:
        print("\nOptions:")
        print("1. Add Student Data")
        print("2. Display Students Table")
        print("3. Search Student by Roll Number")
        print("4. Train Model and Predict Score")
        print("5. Exit")

        choice = input("Select an option (1-5): ")
        
        if choice == "1":
            get_student_data()
        elif choice == "2":
            display_table()
        elif choice == "3":
            roll_number = int(input("Enter Roll Number to search: "))
            search_student_by_roll(roll_number)
        elif choice == "4":
            train_and_predict()
        elif choice == "5":
            print("Exiting the program.")
            break
        else:
            print("Invalid choice. Please select a valid option.")
