# TASK-06
# 🌸 Iris Flower Classification using KNN

This project implements a *K-Nearest Neighbors (KNN) classifier* on the famous *Iris dataset*.  
The goal is to classify iris flowers into three species (Setosa, Versicolor, Virginica) based on their features:  
- Sepal Length  
- Sepal Width  
- Petal Length  
- Petal Width  

---

## 📂 Project Structure
- Iris.csv → Dataset file (must be downloaded and placed in the correct path).  
- knn_iris.py → Python script containing the full implementation.  

---

## ⚙ Requirements
Install the required Python libraries before running the project:

bash
pip install pandas numpy matplotlib seaborn scikit-learn


---

## 🚀 Steps in the Code
1. *Load Dataset*  
   Reads the Iris dataset from a CSV file.  

2. *Data Preparation*  
   - Splits features (X) and target labels (y).  
   - Standardizes features using StandardScaler.  

3. *Train-Test Split*  
   Splits data into training (70%) and testing (30%) sets.  

4. *KNN Classifier*  
   - Uses KNeighborsClassifier with n_neighbors=5.  
   - Trains the model on training data.  

5. *Model Evaluation*  
   - Predicts species for test set.  
   - Prints *accuracy, confusion matrix, and classification report*.  
   - Visualizes the confusion matrix using a heatmap.  

---

## 📊 Example Output
- *Accuracy Score* (varies slightly depending on train-test split)  
- *Confusion Matrix Heatmap*  
- *Classification Report* (precision, recall, F1-score)

---

## 📝 Notes
- Update the dataset path in the code:  
  python
  file_path = r"C:\Users\Divya.p\Downloads\Iris(1).csv"
  
- You can tune the number of neighbors (n_neighbors) to optimize performance.  

---

## 🔮 Future Improvements
- Hyperparameter tuning using GridSearchCV.  
- Compare with other classifiers (Decision Tree, Random Forest, Logistic Regression).  
- Deploy as a simple *Streamlit web app* for interactive predictions.  

---

## 📜 License
This project is open-source and free to use for learning purposes.
