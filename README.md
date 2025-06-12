# AlzheimerPredictor4u
# BI-eksamen 2025

## Project Title  
**AlzheimerPredictor4u**
"Early Detection of Alzheimer’s Disease Using Predictive Analytics"

## Contributors
#### Group 4, l25dat4bi1f 
Business Intelligence 2025  
Copenhagen Business Academy, Lyngby  

GitHub Repository: [AlzheimerPredictor4u_BI_Exam
](https://github.com/FeliciaFavrholdt/AlzheimerPredictor4u_BI_Exam)

### Felicia Favrholdt
- Email: [cph-ff62@cphbusiness.dk](mailto:cph-ff62@cphbusiness.dk)  
- GitHub: [https://github.com/FeliciaFavrholdt](https://github.com/FeliciaFavrholdt)

### Fatima Majid Shamcizadh
- Email: [cph-fs156@cphbusiness.dk](mailto:cph-fs156@cphbusiness.dk)  
- GitHub: [https://github.com/Fati01600](https://github.com/Fati01600)

## GitHub Links  
- **Repository**: [AlzheimerPredictor4u_BI_Exam](https://github.com/FeliciaFavrholdt/AlzheimerPredictor4u_BI_Exam.git)  
- **Streamlit Folder**: Located inside the same repository under /Streamlit_app/

## Raw Dataset Link 
**Alzheimers Disease Dataset:**
https://www.kaggle.com/datasets/rabieelkharoua/alzheimers-disease-dataset/data

## Problem Statement  
*How can we use Business Intelligence and AI techniques to assess the risk of Alzheimer's disease based on demographic and lifestyle factors such as age, gender, health status, and daily habits, in order to support early detection and improve preventive care strategies?”*

## Research Questions  
1. Can we predict the risk of Alzheimer's disease based on demographic and lifestyle factors such as age, gender, physical activity, and diet?
2. Which health and lifestyle features are most predictive of an Alzheimer’s diagnosis?
3. Can we build a predictive dashboard to visualize individual risk levels and support clinical decision-making?

## Motivation 

Our motivation is to help doctors, healthcare and professionals to identify patients who are at high risk of developing Alzheimer’s disease, before the symptoms become severe. 
Alzheimer’s is a progressive condition that affects memory/thinking, and early detection can make a big difference in how the disease is managed, and patients quality of life. In this project, we want to explore how Business Intelligence (BI) and Artificial Intelligence (AI) can be used to analyze real patient data, including age, gender, health conditions, and lifestyle habits like diet and exercise. By finding patterns in the data, we hope to build a smart system that can support earlier diagnosis, so doctors can take action sooner and improve patient outcomes.

## Project Goals

The main goal of this project is to build a simple and practical system that can help doctors and healthcare-staff assess a patient’s risk of developing Alzheimer’s disease. We want to do this by using Business Intelligence (BI) and Artificial Intelligence (AI) techniques on real patient data. 

### Our goals include:

- Creating a machine learning model that can predict the likelihood of an Alzheimer’s diagnosis.
- Identifying which features—such as age, gender, health conditions, and lifestyle habits are most strongly linked to Alzheimer’s risk.
- Designing an interactive dashboard that presents predictions and feature insights in a clear, user-friendly way for clinical use

## Hypotheses
In this project, we aim to uncover the following patterns in the dataset:
- **H1:** Patients over the age of 75 are more likely to be diagnosed with Alzheimer’s than younger individuals
- **H2:** Lower MMSE (Mini-Mental State Exam) scores and higher CDR (Clinical Dementia Rating) scores are strong indicators of Alzheimer’s diagnosis
- H3: Patients who report higher physical activity and better diet quality show lower risk levels for Alzheimer’s disease

## Project Scope and Impact

This project explores how Business Intelligence (BI) and machine learning can help detect Alzheimer’s disease at an early stage.  
We use a real dataset with over 2,000 patients, including details like age, gender, health conditions, and lifestyle habits.  
By analyzing this data, we aim to build a system that helps doctors identify who may be at risk before symptoms become severe.  
The goal is to support faster and more informed decisions in healthcare using data.

### Key Objectives

- Use real patient data to train a model that can predict Alzheimer’s risk.
- Identify which features (age, diet, exercise, test scores) are most important for prediction.
- Clean and prepare the data using BI methods to make it ready for analysis.
- Build an interactive dashboard that presents predictions clearly for healthcare staff.
- Document the full process in Jupyter Notebooks and publish the project on GitHub.

## Expected Outcomes

By the end of the project, we aim to deliver:
- A clean, structured dataset ready for analysis.
- A trained machine learning model for Alzheimer’s risk prediction.
- Saved visualizations and model files for future use.
- A simple Streamlit web app to explore predictions and insights.
- Clear documentation of the problem, the steps taken, and what we learned.
- A complete project repository shared on GitHub.

## Impact and Beneficiaries

This project supports healthcare professionals, such as doctors and nurses, in detecting Alzheimer’s earlier and planning better care.  
It improves decision-making, saves time, and helps patients get support sooner.  
The dashboard could be especially useful in clinics, hospitals, or memory care units working with elderly patients or those showing early signs of cognitive decline.

## Brief Annotation

**1. Which challenge would you like to address?**  
We want to solve the challenge of detecting Alzheimer’s disease early by analyzing patient data such as age, gender, health history, and lifestyle habits. Our goal is to use data to find patterns that show who might be at higher risk, so healthcare professionals can act sooner and provide the right care.
    
**2. Why is this challenge an important or interesting research goal?**  
Alzheimer’s is a serious illness that affects memory and daily life, and it worsens over time. Early detection is key, but it’s not always easy. If we can use data to spot early warning signs, doctors can respond faster, which can help improve quality of life and slow the progression of the disease.    

**3. What is the expected solution your project would provide?**  
We plan to build a machine learning model that uses real patient data to predict a person’s risk of Alzheimer’s. The results will be shown in a simple, visual dashboard that helps doctors and nurses quickly understand the predictions and which factors matter most.
    
**4. What would be the positive impact of the solution, and which category of users could benefit from it?**  
Our solution can support doctors, nurses, and caregivers by giving them better tools to detect Alzheimer’s earlier. It could be used in clinics, hospitals, or memory care units to make smarter decisions, save time, and improve care for patients who need it most.

## Notebooks 
The project is implemented through the 5 modular notebooks, each focused on a specific phase of the BI and AI workflow:

- **01_Problem_Statement_and_Setup**
Define the project scope, research goals, problem formulation, and setup. This includes creating the folder structure, initializing libraries, and documenting the project plan to ensure a clean and reproducible environment.

- **02_Data_Loading_And_Preprocessing**
Load the Alzheimer’s dataset, inspect for missing values, rename columns, and remove duplicates. Perform data cleaning, encode categorical variables, scale numerical values, and handle outliers. Prepare the dataset for modeling by selecting key features related to Alzheimer’s risk.

- **03_Exploratory_Data_Analysis (EDA)**
Explore the data using descriptive statistics and BI visualizations. Generate histograms, boxplots, and correlation heatmaps to understand patterns across age, gender, lifestyle habits, and diagnosis. Identify potential risk indicators.

- **04_Model_Training_and_Evaluation**
Train classification models (Logistic Regression, Decision Trees, Random Forest) to predict Alzheimer’s diagnosis risk. Evaluate model performance using accuracy, precision, recall, confusion matrices, and cross-validation.

- **05_Results_and_Interpretation**
Present the final model results and analyze which features were most important for predictions. Prepare visual outputs for the dashboard and write a summary explaining the model’s performance, strengths, and limitations in a clinical context.



## Streamlit Application
We are building a separate Streamlit web app to make our results easy to explore and understand, even for users without technical knowledge. The app will let doctors and healthcare staff interact with the predictions and visualizations in a simple and clear interface. Users will be able to upload data, view charts, and see model results that show the predicted risk of Alzheimer’s.

**The dashboard** will include visual tools like bar charts, feature importance plots, and summary tables. These elements help users compare different patients, see which features matter most, and understand what drives the predictions. The app will be created using standard Python libraries and will be included in a separate /streamlit_app/ folder in our GitHub repository.

To make the dashboard even more interactive, we will integrate a built-in chatbot using RAG (Retrieval-Augmented Generation). This chatbot will answer questions based on the project data and model results, helping users quickly understand insights and explore explanations, all in natural language. It is designed to support doctors and nurses who need fast, clear answers without diving into the technical details.

## How to run the Streamlit Application 
To run the Streamlit dashboard locally, follow these steps:

1. **Install Python**  
   Download and install Python: [https://www.python.org/downloads/](https://www.python.org/downloads/)

Clone the repository and navigate to the folder

2. **Clone the Repository**
   ```bash
   git clone git@github.com:FeliciaFavrholdt/BI_EXAM.git
   cd Streamlit_app
   ```

Install Required Packages In the terminal, run the following command to install all necessary packages:

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

Once dependencies are installed, launch the app with:

4. **Start the App**
   ```bash
   streamlit run app.py
   ```

This will open the Streamlit web application locally in your browser.

## Execution Plan: BI Sprints

**Sprint 1: Problem Formulation**  
Notebook: 01_Problem_Statement_And_Setup.ipynb
Focus: Define the business case, problem formulation, research questions, and hypotheses. Set up the project environment, folder structure, and GitHub workflow. Document all planning steps to ensure a reproducible project foundation.

**Sprint 2: Data Collection & Preprocessing**  
Notebook: 02_Data_Loading_And_Preprocessing.ipynb 
Focus: Load the Alzheimer’s dataset, inspect and clean the data, handle missing values and outliers, and prepare it for analysis. Apply data transformation, scaling, and encoding to shape the dataset for modeling.   
                                                                                                                                                                                                       
**Sprint 3: Machine Learning & Evaluation**  
Notebooks: 03_Exploratory_Data_Analysis.ipynb, 04_Model_Training_and_Evaluation.ipynb
Focus: Use descriptive statistics and visualizations to explore key patterns and relationships in the data. Then train classification models to predict Alzheimer’s risk, evaluate them using performance metrics, and select the best model.

**Sprint 4: Business Application**  
Notebook: 05_Results_and_Interpretation.ipynb
Streamlit App: streamlit_app/streamlit_app.py
Focus: Build a user-friendly Streamlit dashboard that presents predictions, feature importance, and key visuals. Integrate a chatbot using RAG (Retrieval-Augmented Generation) to allow users to ask questions and receive clear, natural language answers based on the model and data.

## Team Member Engagement 

| Member    | Responsibility                                       | Sprint Phase                   | Deadline          |
|-----------|------------------------------------------------------|--------------------------------|-------------------|
| Fatima & Felicia  | Problem formulation, goals, and setup                | Sprint 1: Problem Formulation  | Wednesday 4/6-25  |
| Fatima & Felicia | Data loading, cleaning, and transformation           | Sprint 2: Data Preparation     | Sunday 8/6-25     |
| Fatima & Felicia   | Model training, tuning, and evaluation               | Sprint 3: Machine Learning     | Thursday 12/6-25  |
| Fatima & Felicia    | Dashboard development and chatbot integration        | Sprint 4: Business Application | Monday 15/6-25    |
| Fatima & Felicia  | Cleanup – refactoring, typos, comments etc.          | Sprint 5: Prepare Handin       | Monday 16/6-25    |
| Fatima & Felicia  | Final checks and delivery                            | Project Deadline               | Tuesday 17/6-25   |

Both team members actively participated in reviewing, testing, and refining the work before each deliverable was uploaded to GitHub or submitted for evaluation.

## Project Directory Structure

Below is an overview of the folder structure used in our project, showing how we separate raw data, models, notebooks, and visual output to keep the project clean and modular:

```


```

## Environment Setup

The project runs in a standard Python environment, suitable for data analysis, machine learning, and web-based dashboards.  
To keep results consistent across systems, all file paths are relative, and setup is handled through utility scripts.

### Libraries Used

The following Python libraries support each part of the workflow:
- **pandas** – Data handling and manipulation with DataFrames.
- **numpy** – Efficient numeric computations.
- **matplotlib** and **seaborn** – Visualization and styling of data.
- **scikit-learn** – Preprocessing, model training, and evaluation.
- **joblib** – Saving and loading trained machine learning models.
- **streamlit** – Creating the interactive web dashboard.

### Development Tools

- **IDE**: Visual Studio Code with Jupyter extension.
- **Environment Management**: Anaconda or pip via requirements.txt.
- **Version Control**: Git with GitHub repository integration.
- **Deployment Tools**: Jupyter Notebooks for analysis and Streamlit for delivery.

### Platform Requirements
To run the project and notebooks successfully, ensure the following software versions are installed:

- **Python** 3.9 or higher  
- **Jupyter Notebook** (or Visual Studio Code with Jupyter extension)  
- **Streamlit** version 1.20 or later  

All notebooks are executed from within the /notebooks/ directory. Output files such as datasets, visualizations, and model summaries are saved to corresponding subdirectories located one level up for consistency across the project.

### Dependency Management

To ensure consistency, all required libraries are listed in a requirements.txt file located in the project root.  
This allows anyone or reviewer to recreate the same environment using a single command. 
The file includes all dependencies needed for notebooks, machine learning, visualizations, and the Streamlit app.

## Project Initialization

To ensure this notebook has access to all necessary paths and tools, we import a helper function init_environment() from utils/setup_notebook.py.  

This function adds project folders (data, reports, plots) to the Python path and sets default styles for visualizations.

#### What init_environment(), does?

As explained before: The init_environment() function performs the following tasks:
1) It applies a consistent visual style using Seaborn and Matplotlib.
2) Verifies that key directories exist (../data, ../models, ../plots, ../reports).
3) Ensures that all notebooks start from a clean and ready-to-use environment.

## How we save histograms, boxplots, and other figures

To ensure consistency, organization, and easy reuse of our visualizations, we created a custom helper function called save_plot() in utils/save_tools.py. This function automatically saves each figure (histograms, boxplots, correlation heatmaps) to the plots/ folder. It also creates a ".txt file" containing a human-readable caption, which we use directly in our Streamlit dashboard.

This process helps us:
- Maintain organized, timestamped visuals across notebooks and sprints.
- Easily load images and captions in Streamlit using st.image().
- Preserve clean notebook outputs (we use plt.close() after saving).

----
## Dataset Information

**Source:**  

Alzheimer’s Disease Dataset by Rabie El Kharoua  
Kaggle URL: https://www.kaggle.com/datasets/rabieelkharoua/alzheimers-disease-dataset/data  
DOI: 10.34740/KAGGLE/DSV/8668279

This dataset includes information from over 2,000 patients and is designed to support research into early detection of Alzheimer’s disease. It contains a wide range of features grouped into categories:

#### Patient Demographics
- **PatientID:** Unique identifier for each individual
- **Age:** Between 60 and 90 years
- **Gender:** 0 = Male, 1 = Female
- **Ethnicity:** 0 = Caucasian, 1 = African American, 2 = Asian, 3 = Other
- **EducationLevel:** 0 = None, 1 = High School, 2 = Bachelor's, 3 = Higher

#### Lifestyle and Habits
- **BMI:** Body Mass Index (15 to 40)
- **Smoking:** 0 = No, 1 = Yes
- **AlcoholConsumption:** Weekly units (0 to 20)
- **PhysicalActivity:** Weekly hours (0 to 10)
- **DietQuality:** Score (0 to 10)
- **SleepQuality:** Score (4 to 10)

#### Medical History
- **FamilyHistoryAlzheimers, CardiovascularDisease, Diabetes, Depression, HeadInjury, Hypertension:** Each coded as 0 = No, 1 = Yes

#### Clinical Measurements
- **SystolicBP, DiastolicBP:** Blood pressure (mmHg)
- **CholesterolTotal, LDL, HDL, Triglycerides:** Cholesterol measures (mg/dL)

#### Cognitive and Functional Assessments
- **MMSE:** Mini-Mental State Exam score (0 to 30), lower values suggest cognitive decline
- **CDR, FunctionalAssessment, ADL:** Functional and cognitive functioning scores
- **MemoryComplaints, BehavioralProblems:** 0 = No, 1 = Yes

#### Symptoms
- **Confusion, Disorientation, PersonalityChanges, DifficultyCompletingTasks, Forgetfulness:** All binary indicators (0 = No, 1 = Yes)

#### Diagnosis
- **Diagnosis:** 0 = No Alzheimer's, 1 = Alzheimer’s diagnosis (Target variable)

**Note:**  
A column named `DoctorInCharge` is included but contains anonymized values ("XXXConfid") and is not used in analysis.

#### Why This Dataset?
This dataset is well-suited for developing machine learning models that assess Alzheimer’s risk. It allows us to explore how demographic, medical, and lifestyle factors interact and how they relate to cognitive conditions.

**Citation:**  
If this dataset is used in academic or public work, it should be cited as:

Rabie El Kharoua (2024). *Alzheimer’s Disease Dataset*. Kaggle. DOI: 10.34740/KAGGLE/DSV/8668279
