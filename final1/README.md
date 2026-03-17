# EduRisk AI - Student Risk Prediction System

An intelligent educational analytics platform that leverages machine learning to predict and identify at-risk students, enabling proactive academic interventions through data-driven insights.

## 🎯 Overview

EduRisk AI uses RandomForest machine learning algorithms to analyze student performance patterns and predict academic risk levels based on attendance rates, academic scores, and study behaviors. The system provides educators with actionable insights through an intuitive web dashboard.

## ✨ Features

- **AI-Powered Risk Prediction**: RandomForest classifier for accurate student risk assessment
- **Real-time Dashboard**: Interactive web interface for viewing student analytics
- **Comprehensive Analytics**: Visualizations of attendance, scores, and performance trends
- **Risk Categorization**: Binary classification system identifying at-risk students
- **Secure Authentication**: User login and session management
- **Data Visualization**: Charts and graphs using Matplotlib and Seaborn
- **Responsive Design**: Modern UI built with Tailwind CSS

## 🛠️ Tech Stack

### Backend
- **Flask** - Python web framework
- **MySQL** - Database for user data and application state
- **Python** - Core programming language

### Machine Learning
- **scikit-learn** - RandomForest classifier and model building
- **pandas** - Data manipulation and analysis
- **numpy** - Numerical computations
- **pickle** - Model persistence and deployment

### Frontend
- **Tailwind CSS** - Utility-first CSS framework
- **HTML5/CSS3** - Modern web standards
- **Font Awesome** - Icon library
- **Google Fonts** - Typography (Inter font family)

### Data Visualization
- **Matplotlib** - Statistical plotting
- **Seaborn** - Enhanced data visualization

## 📋 Prerequisites

- Python 3.7+
- MySQL Server
- pip package manager

## 🚀 Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/edurisk-ai.git
   cd edurisk-ai
   ```

2. **Install dependencies**
   ```bash
   pip install -r htmlnewedu/requirements.txt
   pip install flask mysql-connector-python
   ```

3. **Setup MySQL Database**
   - Create a database named `educational_data`
   - Update database credentials in `htmlnewedu/app.py` (lines 33-37)

4. **Train the ML Model**
   ```bash
   cd htmlnewedu
   python student_risk_prediction.py
   ```

5. **Run the application**
   ```bash
   python app.py
   ```

6. **Access the application**
   - Open your browser and navigate to `http://localhost:5000`

## 📊 Machine Learning Pipeline

### Data Preprocessing
- **Risk Classification**: Binary target variable creation based on:
  - Grade thresholds (D or F grades)
  - Attendance rates (< 75%)
  - Academic performance (Total Score < 50)
- **Feature Selection**: Key academic indicators including attendance, scores, and study hours
- **Feature Engineering**: Optimized input variables for model training

### Model Performance
- **Algorithm**: RandomForest Classifier with 100 estimators
- **Features**: Attendance (%), Total Score, Projects Score, Study Hours per Week
- **Output**: Binary risk prediction with feature importance analysis

## 📁 Project Structure

```
final1/
├── htmlnewedu/
│   ├── app.py                              # Main Flask application
│   ├── student_risk_prediction.py          # ML model training script
│   ├── requirements.txt                    # Python dependencies
│   ├── student_risk_model_v2.pkl          # Trained ML model
│   ├── feature_importances_v2.csv         # Feature importance data
│   ├── Students Performance Dataset.csv    # Training dataset
│   ├── static/
│   │   └── corr_heatmap.png               # Correlation visualization
│   └── templates/                          # HTML templates
│       ├── base.html                       # Base template
│       ├── Dashboard.html                  # Main dashboard
│       ├── login.html                      # Authentication
│       └── ...                             # Other pages
└── README.md
```

## 🔧 Configuration

### Database Setup
Update the MySQL connection parameters in `app.py`:
```python
db = mysql.connector.connect(
    host="localhost",
    user="your_username",
    password="your_password",
    database="educational_data"
)
```

### Model Training
The ML model can be retrained by running:
```bash
python student_risk_prediction.py
```

## 📈 Usage

1. **Login/Signup**: Create an account or login to access the dashboard
2. **Dashboard**: View overall student analytics and risk assessments
3. **Student Details**: Access individual student performance data
4. **Reports**: Generate comprehensive academic reports
5. **Settings**: Configure system preferences

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👥 Authors

- **Your Name** - *Initial work* - [YourGitHub](https://github.com/yourusername)

## 🙏 Acknowledgments

- Educational institutions providing valuable feedback
- Open source community for excellent libraries and frameworks
- Contributors who helped improve the system

## 📞 Contact

- Project Link: [https://github.com/yourusername/edurisk-ai](https://github.com/yourusername/edurisk-ai)
- Email: your.email@example.com

---

⭐ **Star this repository if you found it helpful!**