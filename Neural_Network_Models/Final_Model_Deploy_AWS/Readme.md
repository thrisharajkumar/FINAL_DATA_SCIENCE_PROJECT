FINAL MODEL STRUCTURE + AWS INTEGRATION MAP

Neural_Network_Models/
└── Final_Deploy_AWS/
    ├── Final_Model_With_Deployment.ipynb       ◀️ Main notebook for your thesis
    ├── model_artifacts/                         ◀️ Output from EC2 (downloaded via scp)
    │   ├── final_model.h5                       ✅ Trained ANN model
    │   ├── input_scaler.pkl                     ✅ Saved input scaler
    │   ├── output_scaler.pkl                    ✅ Saved dict of output scalers
    │   ├── training_history.pkl                 ✅ For plotting train/val loss
    │   ├── optuna_best_params.json              ✅ Best hyperparameters used
    │   ├── predictions.csv                      ✅ Actual vs predicted on test/unseen
    │   ├── r2_rmse_table.csv                    ✅ Evaluation metrics table
    │   ├── aws_predictions.csv                  ✅ Predictions from AWS API (optional)
    │
    ├── train_val_loss_curves/                   ✅ 📈 PNG plots
    │   └── train_val_loss_curve.png
    │
    ├── predicted_vs_actual/                     ✅ 📉 Line plots for unseen/test data
    │   └── pulse1_overshoot_plot.png
    │   └── pulse2_undershoot_plot.png
    │
    ├── r2_rmse_tables/                          ✅ 📊 CSVs & plots of metrics
    │   └── r2_bar_chart.png
    │   └── r2_rmse_table.csv
    │
    ├── lambda_function/                         ✅ 📦 Code for AWS Lambda deployment
    │   ├── lambda_function.py
    │   ├── final_model.h5
    │   ├── input_scaler.pkl
    │   ├── output_scaler.pkl
    │   └── requirements.txt
    │
    ├── api_test_request.ipynb                   ✅ 🌐 Send API requests from notebook
    ├── deploy_instructions.md                   ✅ 📄 Step-by-step guide for AWS setup

🔁 HOW THIS WORKS — STEP BY STEP
🔹 Step 1: Run 25% Data Tests (Locally)

Folder: Baseline_ANN/, Attention_ANN/, etc.

You test:

Generalization on unseen MOSFET

Training on 25% balanced data

Log best R² across all outputs

✅ You select the best model type (e.g., Multi-Headed ANN)

🔹 Step 2: Move Best Model to Full Training (AWS EC2)

In EC2:

Run train_final_model.py using full dataset (400K+ rows)

Save .h5, .pkl, predictions, plots, tables

✅ Everything gets saved in ~/final_model_output/

🔹 Step 3: Download from EC2 to Local Project

From your local terminal:

scp -i your-key.pem -r ubuntu@<ec2-ip>:~/final_model_output/ \
Neural_Network_Models/Final_Deploy_AWS/model_artifacts/


✅ Now your local project has the full model results from AWS!

🔹 Step 4: Load All Results in .ipynb

Inside Final_Model_With_Deployment.ipynb, you:

Load the trained model (.h5)

Load predictions and metrics CSVs

Plot training/val loss curves

Compare predicted vs actual

Load AWS predictions via API (optional)

Generate thesis-ready R²/RMSE tables

📘 Thesis Integration

In your .ipynb:

Use ## markdown sections like:

## 1. Final Model Trained on AWS
## 2. R² and RMSE Results
## 3. Generalization on Unseen MOSFET
## 4. Real-Time Inference via AWS Lambda
## 5. Conclusion


In your thesis PDF:

Export key plots (e.g. train_val_loss_curve.png)

Copy summary table from r2_rmse_table.csv

You Are Now Doing:
Phase	Tool	Location
1. Try 25% data with all models	MLP, Attention, Masked	Locally
2. Choose best generalizing model	Based on R²/plots	Locally
3. Train best model on 100% data	Full ANN + Optuna	AWS EC2
4. Save results	.h5, .pkl, plots, CSVs	EC2 then download
Inference (optional)	AWS Lambda + API Gateway	Notebook
5. Final notebook	Final_Model_With_Deployment.ipynb	Locally, fully explained