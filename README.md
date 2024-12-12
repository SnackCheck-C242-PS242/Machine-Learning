## SnackCheck

# Machine Learning

The [**SnackCheck Dataset**](https://www.kaggle.com/datasets/yashkaggle27/nutrition-dataset-for-healthy-food-prediction) which includes nutritional information per 100 grams for various snack categories. The dataset has undergone the following stages:
- Gathering data.
- Assessing data.
- Cleaning data.
- Exploratory Data Analysis (EDA)

**Dataset Information**

- fat 100g, saturated fat 100g, carbohydrates 100g, sugars 100g, fiber 100g, proteins 100g, sodium 100g.


## System Requirements
- Python 3.7 or later.
- TensorFlow 2.x
- Pandas, NumPy, Matplotlib for data analysis


## Steps for Model Training

1. Download the datasets from the following link: [**SnackCheck Dataset**](https://www.kaggle.com/datasets/yashkaggle27/nutrition-dataset-for-healthy-food-prediction)
2. Pre-processing the data and clean it up by separating the data into features (X) and label (y) then split into training and test sets.
![{147F3EA1-1083-48C0-A811-56DFE93EF333}](https://github.com/user-attachments/assets/07e6a6ab-6766-4247-8ce8-c0e3187106ae)

3. Building the model using a TensorFlow sequential model with the following architecture:
![{E85D3437-3336-4D03-BC49-164450062E97}](https://github.com/user-attachments/assets/81525983-a4d4-4af5-af13-a10f61d405d0)

4. The model is compiled using the Adam optimizer and binary crossentropy loss. Training is performed for 50 epochs with a batch size of 32, using a validation split of 20%.
![{5469C570-3390-4F1E-8798-C157DBB30D5F}](https://github.com/user-attachments/assets/356bc3df-ecbb-44f7-a19d-a6522874017f)

5. The trained model is saved in HDF5 format (nutrisnack_model.h5).
7. Predict the healthiness of snacks and categorize their nutritional values based on thresholds.
8. Converted to TensorFlow Lite format (model.tflite) for deployment.
9. Using Flask API for predictions


## How to Use
1. Clone this repository

```
git clone https://github.com/username/SnackCheck.git
```

2. Install all library

```
pip install -r requirements.txt
```

3. Run SnackCheck.ipynb
4. Run app.py to start the API
```
python app.py
```
5. Make a Prediction:
Send a POST request to http://127.0.0.1:8080/predict with JSON input containing nutritional values.

