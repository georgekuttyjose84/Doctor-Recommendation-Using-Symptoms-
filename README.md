# Doctor Recommendation System

This repository contains a project for predicting diseases based on symptoms and recommending the appropriate department for medical consultation. The system uses a machine learning model to make predictions and provides an interactive GUI for users to input symptoms and receive results.

---

## Project Structure

### 1. **Python Script** - `Doctor_Recommendation.py`
- Implements the disease prediction system using a Naive Bayes model.
- Key functionalities:
  - **Data Loading & Training**: Trains a Naive Bayes classifier using medical datasets.
  - **GUI Integration**: Built with `Tkinter` for user-friendly symptom selection and results display.
  - **Disease-to-Department Mapping**: Links diseases to medical specialties based on a predefined mapping.

### 2. **Disease to Department Mapping** - `Disease_to_Department_Table.pdf`
- Contains a comprehensive table mapping diseases to the respective medical departments, ensuring the user receives appropriate doctor recommendations.

### 3. **Model Evaluation Notebook** - `Checking Different Model Accuracy.ipynb`
- Explores and evaluates the performance of different machine learning models for disease prediction.
- Helps compare the Naive Bayes model's effectiveness against alternatives like Decision Trees, Random Forests, and others.

---

## Features
- **Symptom-based Prediction**: Enter symptoms to get a predicted disease and recommended doctor.
- **Expandable Interface**: Add more symptoms dynamically.
- **Model Training**: A Naive Bayes classifier trained on medical data.
- **Comprehensive Mapping**: Includes 40+ diseases mapped to relevant specialties.

---

## Setup and Usage

### Prerequisites
- Python 3.7 or above
- Required Python libraries: `tkinter`, `pandas`, `numpy`, `scikit-learn`

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/your_username/doctor-recommendation.git
   cd doctor-recommendation
   ```
2. Install required libraries:
   ```bash
   pip install -r requirements.txt
   ```

### Running the Application
1. Ensure the training dataset (`Datasets/Training.csv`) is available in the correct path.
2. Run the `Doctor_Recommendation.py` script:
   ```bash
   python Doctor_Recommendation.py
   ```
3. Interact with the GUI to select symptoms and get predictions.

---

## How It Works
1. **User Input**: Select symptoms via dropdowns in the GUI.
2. **Prediction**:
   - The selected symptoms are converted into a format compatible with the trained model.
   - The model predicts the most probable disease.
3. **Recommendation**: Based on the predicted disease, the system recommends a specialist department for consultation.

---

## Data Insights
- Training data includes symptoms, diseases, and mappings.
- Diseases covered include common conditions like **Allergy, Diabetes, Migraine** as well as more specific ones like **GERD, Hepatitis**.

---

## Future Improvements
- Add more diseases and symptoms to the dataset.
- Integrate alternative machine learning models for better accuracy.
- Provide additional patient instructions based on predictions.
- Develop a web-based version of the application.

---

## Contributors
- **Georgekutty Jose**
  - Email: [georgekuttyjose84@gmail.com](mailto:georgekuttyjose84@gmail.com)
  - LinkedIn: [georgekutty108](https://www.linkedin.com/in/georgekutty108)

---