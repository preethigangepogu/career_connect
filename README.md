# CareerConnect - Job Portal System

A full-stack job portal where companies can post jobs and candidates can apply.
The system automatically evaluates how well a candidate matches a job and can
generate job descriptions and candidate summaries.

## Project Folder Structure

```
CareerConnect/
├── Frontend/
│   ├── index.html           - Home page
│   ├── css/style.css        - Stylesheet
│   ├── js/utils.js          - Shared JS functions
│   └── pages/
│       ├── login.html       - Login page
│       ├── register.html    - Register page
│       ├── jobs.html        - Browse jobs (public)
│       ├── candidate.html   - Candidate dashboard
│       └── recruiter.html   - Recruiter dashboard
│
├── Backend/
│   └── app.py               - Main server (Flask)
│                               (careerconnect.db is created automatically on first run)
│
├── ML/
│   ├── train_model.py        - Model training script
│   ├── Model_Training.ipynb  - Jupyter notebook (data prep, training, evaluation)
│   ├── dataset.csv           - Training dataset
│   ├── rf_model.pkl          - Trained model
│   ├── edu_encoder.pkl       - Encoder file
│   └── label_encoder.pkl     - Encoder file
│
├── Project_Documentation.docx - Problem statement, approach, architecture, tools
├── README.md
└── requirements.txt
```

## How to Run

### Step 1 - Install required libraries
```
pip install flask scikit-learn pandas numpy joblib
```

### Step 2 - Train the model (only once)
```
cd CareerConnect/ML
python train_model.py
```
This creates `rf_model.pkl`, `edu_encoder.pkl`, `label_encoder.pkl`, and `dataset.csv`
inside the ML folder. You should see an accuracy of about 96-97%.

### Step 3 - Start the server
```
cd ../Backend
python app.py
```

### Step 4 - Open in browser
Go to: **http://127.0.0.1:5000**

---

## How to Test

1. Open http://127.0.0.1:5000
2. Click Register -> create a Recruiter account (fill in company name)
3. Login as Recruiter -> Post a Job -> try "Auto-Generate Description"
4. Register a Candidate account -> Login -> Apply for the job
5. See the compatibility result and score after applying
6. Login back as Recruiter -> Applicants tab -> view candidates ranked by score
7. Click "View" on a candidate -> "Generate Candidate Summary"
8. Update application status to Shortlisted / Hired
9. Check the Overview tab for dashboard stats

## Model Training Notebook

Open `ML/Model_Training.ipynb` in Jupyter Notebook to see:
- Dataset creation
- Data preprocessing and encoding
- Model training (Random Forest)
- Model evaluation (accuracy, confusion matrix, classification report)
- Feature importance
- Saving the trained model

## Tech Stack

- Frontend: HTML, CSS, JavaScript
- Backend: Python, Flask
- Database: SQLite
- Model: Random Forest (scikit-learn)
- Description Generator: Built-in template-based module (no external API needed)
