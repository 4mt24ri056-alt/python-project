# Spam Detection System 📩

A machine learning web application that classifies text messages as **Spam** or **Not Spam** using Natural Language Processing (NLP) and the Multinomial Naive Bayes algorithm.

## 🚀 Features

- Classifies messages as Spam or Not Spam.
- Uses machine learning for text classification.
- Converts text into numerical features using CountVectorizer.
- Removes duplicate records from the dataset.
- Displays model accuracy on test data.
- Provides an interactive web interface using Streamlit.
- Validates user input before making predictions.

## 🛠️ Technologies Used

- **Python** – Programming language
- **Pandas** – Dataset handling and preprocessing
- **Scikit-learn** – Machine learning and text vectorization
- **CountVectorizer** – Converts text into numerical features
- **Multinomial Naive Bayes** – Classification algorithm
- **Streamlit** – Web application interface

## 📂 Project Structure

```text
python-project/
├── spam_app.py       # Streamlit application
├── spam.csv          # Dataset (required)
└── README.md         # Project documentation
```

## ⚙️ Installation

### 1. Clone the Repository

```bash
git clone https://github.com/4mt24ri056-alt/python-project.git
```

### 2. Navigate to the Project Folder

```bash
cd python-project
```

### 3. Install Dependencies

```bash
pip install pandas scikit-learn streamlit
```

## ▶️ How to Run

1. Ensure that `spam.csv` is available in the project directory.
2. Update the dataset path in `spam_app.py` if necessary.
3. Run the application:

```bash
python -m streamlit run spam_app.py
```

4. Open the local URL displayed in the terminal.
5. Enter a message and click **Validate** to see the prediction.

## 🧠 How It Works

1. **Data Loading:** Loads messages and their categories from a CSV dataset.
2. **Preprocessing:** Removes duplicate records and standardizes category labels.
3. **Data Splitting:** Splits the dataset into training and testing sets.
4. **Feature Extraction:** Uses CountVectorizer to convert text into numerical features.
5. **Model Training:** Trains a Multinomial Naive Bayes classifier.
6. **Prediction:** Classifies user-entered messages as Spam or Not Spam.
7. **Evaluation:** Displays accuracy on the test dataset.

## 📊 Machine Learning Algorithm

The project uses **Multinomial Naive Bayes**, a probabilistic classification algorithm commonly used for text classification tasks such as spam detection.

## 🔮 Future Enhancements

- Improve model performance through text preprocessing.
- Display precision, recall, and F1-score.
- Add a confusion matrix for model evaluation.
- Deploy the application online.
- Improve the user interface and prediction results.
  
## output

### 
![not spam](<img width="1920" height="1080" alt="Screenshot 2026-05-22 222646" src="https://github.com/user-attachments/assets/287f24d7-c886-470f-bee4-453e96eb100f" />
)

### 
![Spam](<img width="1920" height="1080" alt="Screenshot 2026-05-22 222620" src="https://github.com/user-attachments/assets/ab2d03df-407e-4215-b887-bdfbf6d25c22" />
)


## 👨‍💻 Author

**4mt24ri056-alt**

## 📄 License

This project is intended for educational and learning purposes.
