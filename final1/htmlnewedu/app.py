# ✅ Updated app.py with safe dataset path handling
from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify
import mysql.connector
from mysql.connector import Error
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import pickle
import json
import os
import traceback

# Debug prints (remove later if not needed)
print("🟡 Current Working Directory:", os.getcwd())
print("📄 Files in this directory:", os.listdir())

# Load the trained model or train a new one
MODEL_PATH = 'student_risk_model_v2.pkl'
DATASET_NAME = 'Students Performance Dataset.csv'
DATASET_PATH = os.path.join(os.path.dirname(__file__), DATASET_NAME)

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file {MODEL_PATH} not found. Please train the model first.")
else:
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'

# Database connection
try:
    db = mysql.connector.connect(
        host="localhost",
        user="root",
        password="Rajani82",
        database="educational_data"
    )
    print("✅ MySQL connection established.")
except Error as e:
    print("❌ Error connecting to MySQL:", e)

@app.route('/')
def index():
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']

        cursor = db.cursor(dictionary=True)
        query = "SELECT * FROM users WHERE email = %s AND password = %s"
        cursor.execute(query, (email, password))
        user = cursor.fetchone()
        cursor.close()

        if user:
            session['user_id'] = user['id']
            session['email'] = user['email']
            session['name'] = user.get('name', 'John Doe')
            session['photo'] = user.get('photo', 'https://randomuser.me/api/portraits/men/32.jpg')
            return redirect(url_for('home'))
        else:
            flash("Invalid credentials.", "error")
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.clear()
    flash("You have been logged out.", "info")
    return redirect(url_for('login'))

@app.route('/home')
def home():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    return render_template('home.html', name=session.get('name'), email=session.get('email'), photo=session.get('photo'))

@app.route('/profile')
def profile():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    return render_template('profile.html', name=session.get('name'), email=session.get('email'), photo=session.get('photo'))

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        confirm_password = request.form['confirm_password']

        if password != confirm_password:
            flash("Passwords do not match.")
            return render_template('signup.html')

        try:
            cursor = db.cursor(dictionary=True)
            cursor.execute("SELECT * FROM users WHERE email = %s", (email,))
            if cursor.fetchone():
                flash("Email already registered.")
                return render_template('signup.html')
            cursor.execute("INSERT INTO users (email, password) VALUES (%s, %s)", (email, password))
            db.commit()
            flash("Account created. Please login.")
            return redirect(url_for('login'))
        except Error as e:
            print("Database error:", e)
            flash("Error occurred. Try again.")
    return render_template('signup.html')

# Use new feature list for all predictions
features = ['Attendance (%)', 'Total_Score', 'Projects_Score', 'Study_Hours_per_Week']

@app.route('/dashboard')
def dashboard():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    df = pd.read_csv(DATASET_PATH)
    X = df[features]
    predictions = model.predict_proba(X)[:, 1]
    df['Risk_Score'] = predictions
    df['At_Risk'] = (
        (df['Grade'].isin(['D', 'F'])) |
        (df['Attendance (%)'] < 75) |
        (df['Total_Score'] < 50)
    ).astype(int)
    total_students = len(df)
    at_risk_count = df['At_Risk'].sum()
    risk_percentage = (at_risk_count / total_students) * 100
    top_risk_students = df.sort_values('Risk_Score', ascending=False).head(5)
    # Load feature importances from feature_importances_v2.csv
    import os
    importance_path = os.path.join(os.path.dirname(__file__), 'feature_importances_v2.csv')
    if os.path.exists(importance_path):
        imp_df = pd.read_csv(importance_path)
        feature_importances = imp_df.to_dict('records')
    else:
        feature_importances = []
    return render_template('Dashboard.html',
                           total_students=total_students,
                           at_risk_count=at_risk_count,
                           risk_percentage=risk_percentage,
                           top_risk_students=top_risk_students.to_dict('records'),
                           feature_importances=feature_importances)

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        data = request.json
        feature_vector = [float(data.get(feature, 0)) for feature in features]
        risk_score = model.predict_proba([feature_vector])[0][1]
        return jsonify({
            'risk_score': float(risk_score),
            'prediction': 'At Risk' if risk_score >= 0.5 else 'Not At Risk'
        })

@app.route('/students')
def students():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    df = pd.read_csv(DATASET_PATH)
    X = df[features]
    predictions = model.predict_proba(X)[:, 1]
    df['Risk_Score'] = predictions
    df['Prediction'] = model.predict(X)
    df['Prediction'] = df['Prediction'].map({1: 'At Risk', 0: 'Not At Risk'})
    students_data = df[['Student_ID', 'First_Name', 'Last_Name', 'Department',
                        'Attendance (%)', 'Total_Score', 'Projects_Score',
                        'Study_Hours_per_Week', 'Risk_Score',
                        'Prediction', 'Grade']].to_dict('records')
    return render_template('Students.html', students=students_data)

@app.route('/reports')
def reports():
    import pandas as pd
    import numpy as np
    import os
    df = pd.read_csv(DATASET_PATH)
    # Use new At_Risk definition for all analytics
    df['At_Risk'] = (
        (df['Grade'].isin(['D', 'F'])) |
        (df['Attendance (%)'] < 75) |
        (df['Total_Score'] < 50)
    ).astype(int)
    X = df[features]
    if 'Risk_Score' not in df.columns:
        try:
            df['Risk_Score'] = model.predict_proba(X)[:, 1]
        except Exception as e:
            df['Risk_Score'] = df['At_Risk']
    # Load feature importances from feature_importances_v2.csv
    feature_importance_data = {}
    importance_path = os.path.join(os.path.dirname(__file__), 'feature_importances_v2.csv')
    if os.path.exists(importance_path):
        imp_df = pd.read_csv(importance_path)
        feature_importance_data = {
            "labels": imp_df['feature'].tolist(),
            "data": imp_df['importance'].round(4).tolist()
        }

    # a. Risk Pie Data
    at_risk_count = int((df['At_Risk'] == 1).sum())
    not_at_risk_count = int((df['At_Risk'] == 0).sum())
    risk_pie_data = {
        "labels": ["At Risk", "Not At Risk"],
        "data": [at_risk_count, not_at_risk_count]
    }

    # b. Department-wise Risk %
    dept_risk = df.groupby('Department')['At_Risk'].mean().reset_index()
    dept_risk_data = {
        "labels": dept_risk['Department'].astype(str).tolist(),
        "data": (dept_risk['At_Risk'] * 100).round(1).tolist()
    }

    # c. Risk vs Attendance
    risk_attendance_data = {
        "x": df['Attendance (%)'].astype(float).tolist(),
        "y": df['Risk_Score'].astype(float).tolist()
    }

    # d. Risk vs Study Hours
    risk_studyhours_data = {
        "x": df['Study_Hours_per_Week'].astype(float).tolist(),
        "y": df['Risk_Score'].astype(float).tolist()
    }

    # e. Average Risk by Grade
    grade_risk = df.groupby('Grade')['Risk_Score'].mean().reset_index()
    grade_risk_data = {
        "labels": grade_risk['Grade'].astype(str).tolist(),
        "data": grade_risk['Risk_Score'].round(3).tolist()
    }

    # f. Top 10 At-Risk Students
    top10 = df.sort_values('Risk_Score', ascending=False).head(10)
    top_risk_students = [
        {
            "name": f"{row.First_Name} {row.Last_Name}",
            "id": row.Student_ID,
            "score": round(row.Risk_Score, 3)
        }
        for row in top10.itertuples()
    ]

    # g. Risk by Sleep Hours (bucketed)
    bins = [0, 5, 6, 7, 24]
    labels = ['<5', '5-6', '6-7', '7+']
    df['Sleep_Bucket'] = pd.cut(df['Sleep_Hours_per_Night'], bins=bins, labels=labels, right=False)
    sleep_risk = df.groupby('Sleep_Bucket')['Risk_Score'].mean().reset_index()
    sleep_risk_data = {
        "labels": sleep_risk['Sleep_Bucket'].astype(str).tolist(),
        "data": sleep_risk['Risk_Score'].round(3).tolist()
    }

    # Department-wise Avg Study Hours
    if 'Department' in df.columns and 'Study_Hours_per_Week' in df.columns:
        dept_study = df.groupby('Department')['Study_Hours_per_Week'].mean().reset_index()
        dept_study_hours_data = {
            "labels": dept_study['Department'].astype(str).tolist(),
            "data": dept_study['Study_Hours_per_Week'].round(2).tolist()
        }
    else:
        dept_study_hours_data = {}

    # Grade Distribution
    if 'Grade' in df.columns:
        grade_counts = df['Grade'].value_counts().sort_index()
        grade_dist_data = {
            "labels": grade_counts.index.tolist(),
            "data": grade_counts.values.tolist()
        }
    else:
        grade_dist_data = {}

    # Average Risk by Grade (duplicate for template compatibility)
    avg_risk_by_grade_data = grade_risk_data

    # Correlation Heatmap (optional, pass None if not generated)
    corr_heatmap_path = None

    # Attendance Buckets vs Average Risk Score (fixed logic)
    attendance_bins = [0, 60, 70, 80, 90, 100]
    attendance_labels = ['<60%', '60–70%', '70–80%', '80–90%', '90–100%']
    df['Attendance_Bucket'] = pd.cut(df['Attendance (%)'], bins=attendance_bins, labels=attendance_labels, right=False, include_lowest=True)
    att_bucket = df.groupby('Attendance_Bucket', observed=True)['Risk_Score'].mean().reindex(attendance_labels)
    attendance_bucket_data = {
        "labels": attendance_labels,
        "data": att_bucket.round(3).fillna(0).tolist()
    }

    # Study Hours Buckets vs At-Risk Percentage (fixed logic)
    study_bins = [0, 3, 5, 8, np.inf]
    study_labels = ['<3 hrs', '3–5 hrs', '5–8 hrs', '8+ hrs']
    df['Study_Hours_Bucket'] = pd.cut(df['Study_Hours_per_Week'], bins=study_bins, labels=study_labels, right=False, include_lowest=True)
    study_bucket = df.groupby('Study_Hours_Bucket', observed=True)['At_Risk'].mean().reindex(study_labels)
    study_hour_risk_data = {
        "labels": study_labels,
        "data": (study_bucket * 100).round(1).fillna(0).tolist()
    }

    return render_template(
        'Reports.html',
        risk_pie_data=risk_pie_data,
        dept_risk_data=dept_risk_data,
        grade_dist_data=grade_dist_data,
        feature_importance_data=feature_importance_data,
        avg_risk_by_grade_data=avg_risk_by_grade_data,
        top_risk_students=top_risk_students,
        sleep_risk_data=sleep_risk_data,
        dept_study_hours_data=dept_study_hours_data,
        grade_risk_data=grade_risk_data,
        attendance_bucket_data=attendance_bucket_data,
        study_hour_risk_data=study_hour_risk_data,
        corr_heatmap_path=corr_heatmap_path
    )

@app.route('/settings')
def settings():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    return render_template('Settings.html')

@app.route('/help_page')
def help_page():
    return render_template('help.html')

@app.route('/student/<student_id>')
def student_details(student_id):
    if 'user_id' not in session:
        return redirect(url_for('login'))

    df = pd.read_csv(DATASET_PATH)
    student = df[df['Student_ID'] == student_id]
    if student.empty:
        return "Student not found", 404
    student = student.iloc[0]

    student_data = {
        'id': student['Student_ID'],
        'name': f"{student['First_Name']} {student['Last_Name']}",
        'department': student['Department'],
        'attendance': student['Attendance (%)'],
        'midterm': student['Midterm_Score'],
        'final': student['Final_Score'],
        'assignments': student['Assignments_Avg'],
        'quizzes': student['Quizzes_Avg'],
        'participation': student['Participation_Score'],
        'projects': student['Projects_Score'],
        'study_hours': student['Study_Hours_per_Week'],
        'stress_level': student['Stress_Level (1-10)'],
        'sleep_hours': student['Sleep_Hours_per_Night'],
        'grade': student['Grade']
    }

    feature_vector = [student[feature] for feature in features]
    risk_score = model.predict_proba([feature_vector])[0][1]
    prediction = 'At Risk' if risk_score >= 0.5 else 'Not At Risk'

    return render_template('StudentDetails.html',
                           student=student_data,
                           risk_score=risk_score,
                           prediction=prediction)

@app.route('/predict_student/<student_id>', methods=['GET'])
def predict_student(student_id):
    df = pd.read_csv(DATASET_PATH)
    student = df[df['Student_ID'] == student_id]
    if student.empty:
        return jsonify({'error': 'Student not found'}), 404
    student = student.iloc[0]
    features = ['Attendance (%)', 'Total_Score', 'Projects_Score', 'Study_Hours_per_Week']
    feature_vector = [student[feature] for feature in features]
    risk_score = model.predict_proba([feature_vector])[0][1]
    prediction = 'At Risk' if risk_score >= 0.5 else 'Not At Risk'
    student_data = {
        'Student_ID': student['Student_ID'],
        'Full_Name': f"{student['First_Name']} {student['Last_Name']}",
        'Department': student['Department'],
        'Attendance': student['Attendance (%)'],
        'Project_Score': student['Projects_Score'],
        'Total_Score': student['Total_Score'],
        'Study_Hours_per_Week': student['Study_Hours_per_Week'],
        'Grade': student['Grade'],
        'Risk_Score': risk_score,
        'Prediction': prediction
    }
    return jsonify(student_data)

@app.route('/predict_department/<dept_name>', methods=['GET'])
def predict_department(dept_name):
    try:
        if not dept_name or dept_name.strip() == '':
            return jsonify({'error': 'Invalid department name.'}), 400
        df = pd.read_csv(DATASET_PATH)
        features = ['Attendance (%)', 'Total_Score', 'Projects_Score', 'Study_Hours_per_Week']
        if 'Department' not in df.columns:
            return jsonify({'error': 'Department column missing in dataset.'}), 400
        dept_df = df[df['Department'] == dept_name]
        if dept_df.empty:
            return jsonify({'error': 'No students found for this department.'}), 400
        for col in features:
            if col not in dept_df.columns:
                return jsonify({'error': f'Missing feature column: {col}'}), 400
        X = dept_df[features]
        if X.isnull().any().any():
            return jsonify({'error': 'Missing values in feature columns.'}), 400
        try:
            predictions = model.predict_proba(X)[:, 1]
        except Exception as e:
            print('Model prediction error:', e)
            import traceback
            traceback.print_exc()
            return jsonify({'error': 'Model prediction failed.'}), 400
        dept_df = dept_df.copy()
        dept_df['Risk_Score'] = predictions
        dept_df['Prediction'] = model.predict(X)
        dept_df['Prediction'] = dept_df['Prediction'].map({1: 'At Risk', 0: 'Not At Risk'})
        total_students = int(len(dept_df))
        at_risk_df = dept_df[dept_df['Prediction'] == 'At Risk']
        at_risk_count = int(len(at_risk_df))
        risk_percentage = float((at_risk_count / total_students) * 100 if total_students > 0 else 0)
        # Prepare all at-risk students
        at_risk_students = []
        for row in at_risk_df.itertuples(index=False):
            at_risk_students.append({
                'Student_ID': str(row.Student_ID),
                'Full_Name': f"{row.First_Name} {row.Last_Name}",
                'Department': str(row.Department),
                'Risk_Score': float(row.Risk_Score),
                'Grade': str(row.Grade)
            })
        return jsonify({
            'total_students': total_students,
            'at_risk_count': at_risk_count,
            'risk_percentage': risk_percentage,
            'at_risk_students': at_risk_students
        })
    except Exception as e:
        print('Error in /predict_department:', e)
        import traceback
        traceback.print_exc()
        return jsonify({'error': 'Internal server error: ' + str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
