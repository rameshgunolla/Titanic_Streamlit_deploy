# Titanic Survival Prediction 🚢

A simple end-to-end machine learning project to predict Titanic passenger survival using **Scikit-Learn** and **Streamlit**.  
This project demonstrates:
- Data preprocessing and feature engineering
- Model training and saving
- Interactive web app for prediction using Streamlit

---

## 📁 Project Structure

Titanic_streamlit_deploy/
│
├── app.py # Streamlit app for interactive prediction
├── data_load.py # Data loading and preprocessing script
├── train.py # Model training script
├── model/ # Directory for saved model and data files
│ ├── preprocessor.joblib
│ ├── model.joblib
│ ├── X_train.npy
│ ├── y_train.npy
│ ├── X_test.npy
│ ├── y_test.npy
├── requirements.txt # Python dependencies
├── .gitignore # Git ignore file
├── README.md # Project documentation


---

## ⚙️ Setup Instructions

1️⃣ **Clone the repository**

```bash
git clone https://github.com/your-username/Titanic_streamlit_deploy.git
cd Titanic_streamlit_deploy

2️⃣ Create a virtual environment (optional but recommended)

python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

3️⃣ Install dependencies

pip install -r requirements.txt

🚂 How to Run
1️⃣ Preprocess the data

python data_load.py
This will:

Load Titanic dataset (from seaborn)

Preprocess & save preprocessor

Save processed train/test sets

2️⃣ Train the model

python train.py --model-output-path model/

This will:

Train a Random Forest model

Save the trained model to model/

3️⃣ Launch the Streamlit app
streamlit run app.py
Open the provided local URL in your browser.
Fill the passenger details → click Predict → see the survival prediction!

📌 Requirements
Python 3.8+

pandas, numpy, scikit-learn, seaborn, streamlit, joblib

💡 Notes
The dataset is loaded directly from seaborn, so no external CSV needed.

Models and preprocessors are saved in the model/ directory.

You can easily extend this app for cloud deployment (e.g., GCP, AWS, Heroku).

📜 License
This project is for educational/demo purposes.
Feel free to fork and adapt!