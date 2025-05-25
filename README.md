# E-commerce Fraud Detection using Behavioral Machine Learning

## Project Overview

This project aims to develop a machine learning model to detect fraudulent e-commerce transactions, drawing inspiration from industry leader Riskified., or a similar version was used for IP geolocation).
*   **Target Variable:** `class` (1 for fraud, 0 for legitimate). The dataset is imbalanced with approximately 9.36% fraudulent transactions.

*Data files: The focus is on engineering features from transactional and contextual data that can act as proxies for behavioral signals, enabling the model to distinguish are not included in this repository due to size. Please download them from the provided Kaggle link.*

## Methodology

 between legitimate and fraudulent activities. The primary goal was to optimize for high precison of fraudulent transactions (as what Riskified also does)  while understanding the inherent recall tradeoff. The project followed a structured data science workflow:

1.  **Data Loading & Initial EDA:** Understanding data structure, distributions-offs.

This was developed as part of the Financial Data Analytics course.

**Dataset Used:** 'https://www.kaggle.com/datasets/vbinh002/fraud-ecommerce/code'.
2.  **Extensive Feature Engineering:** This was a critical step to create behavioral signalsgle Dataset. Key engineered features include:
    *   **IP-to-Country Mapping:** Adding geographical context to transactions.
    *kaggle.com/datasets/vbinh002/fraud-ecommerce) and a supplementary IP-to-   **Advanced Time Analysis:**
        *   `seconds_since_signup`: Time elapsed between user signup and purchase.
        *   Cyclical Encoding: For purchase/signup hour, day of week, and day of month toCountry mapping table.

## Key Features & Methodology

1.  **Data Ingestion & EDA:** Loaded transaction data (150k capture temporal patterns.
        *   Categorical time features like `period_day_cat` and flags like `quick_1k entries) and IP geolocation data. Performed initial EDA to understand class imbalance (approx. 9.4% fraud)purchase_flag`.
    *   **Behavioral/Device/Contextual Features:**
        *   `countries_from and feature distributions.
2.  **Extensive Feature Engineering:**
    *   **IP-to-Country_device`: Number of distinct countries associated with a device ID.
        *   `device_id_freq`: Mapping:** Enriched transactions with geographical context.
    *   **Time-based Features:** `seconds_since_ Frequency of a given device ID in the dataset.
        *   `risk_country_cat`: Categorizing IPsignup`, cyclical encoding of hour/day/month, `quick_purchase_flag`, `period_of_day`.
 countries based on historical fraud rates.
        *   `age_cat`: Binning user age.
        *   `first_purchase_flag`: Indicating if it's the user's first transaction.
    *   **Logarithmic    *   **Behavioral/Device Proxies:** `countries_from_device` (number of countries associated with a device_id), `device_id_freq` (rarity of device), `risk_country_cat` (country Transformations & Interactions:** For skewed numerical data (e.g., `purchase_value_log`) and creating interaction terms (e.g., `purchase_value_log_per_time_since_signup_seconds_log`).
     risk based on historical fraud).
    *   **Value Transformations & Interactions:** Log transformations for skewed data (`purchase_value`,*   This resulted in a dataset with **44 engineered features.**
3.  **Initial Model Bake-off:** `seconds_since_signup`), and interaction terms (e.g., `purchase_value_per_time_since Evaluated 10 different classification algorithms (including Logistic Regression, Decision Tree, Random Forest, XGBoost, LightGBM,_signup`).
    *   This resulted in 44 engineered features.
3.  **Model Bake-off:** CatBoost, KNN, Naive Bayes, AdaBoost, Linear SVM) using appropriate class weighting strategies to establish baseline performances Evaluated 10 different classification algorithms (including Logistic Regression, Decision Tree, RandomForest, XGBoost, LightGBM, Cat. Tree-based ensembles like XGBoost, LightGBM, and CatBoost showed strong initial results.
4.  **FeatureBoost, KNN, Naive Bayes, AdaBoost, SVM) using both SMOTE and Class Weighting strategies for imbalance Selection:** Employed XGBoost feature importances with `SelectFromModel` (using a "median" threshold) to reduce.
5.  **Feature Selection:** Utilized XGBoost feature importances with `SelectFromModel` to reduce the feature set to the **top 25 most impactful features**, balancing model performance with complexity.
6.  **Final Model Optimization & Evaluation:**
    *   Focused on **XGBoost** as the champion model.
     dimensionality to the top 25 most impactful features.
After comprehensive feature engineering, model bake-offs, and feature selection (resulting in 25 key features), the final models were trained and evaluated. The **Optimized_XGBoost** model, using a **ClassWeight** strategy (specifically `scale_pos_weight` to handle imbalance), emerged as the champion when optimizing for the highest F1-score for fraud detection.

The optimal F1-score was achieved by adjusting the decision threshold to **0.82**. At this operating point, the champion model demonstrated the following performance on the hold-out test set:

*   **Fraud Recall: 53.22%** (The model identified over half of the actual fraudulent transactions.)
*   **Fraud Precision: 99.01%** (Of transactions flagged as fraud by the model at this threshold, 99% were actually fraudulent, indicating very few false positives among those flagged.)
*   **Fraud F1-Score: 69.23%** (A strong balance between precision and recall, which was the optimization target.)
*   **Overall ROC AUC: 0.8387** (This threshold-independent metric indicates good overall model discriminative power.)

**Confusion Matrix for Optimized_XGBoost (Threshold = 0.82):**
*   True Positives (Fraud caught): 1506
*   False Positives (Legitimate flagged as Fraud): 15
*   False Negatives (Fraud missed): 1324
*   True Negatives (Legitimate correctly identified): 27378

This high-precision outcome (99.01%) at the F1-optimized threshold means the model is very reliable when it flags a transaction as fraudulent, leading to efficient use of resources for reviewing flagged cases. However, the recall of 53.22% indicates that a notable portion of fraudulent transactions would still go undetected with this specific F1-optimized configuration.

For comparison, at the default 0.5 threshold, this same XGBoost model achieved:
*   Fraud Recall: 71.27%
*   Fraud Precision: 54.03%
*   Fraud F1-Score: 61.47%

This highlights the critical role of threshold selection in tailoring model behavior to specific operational goals (e.g., maximizing F1, maximizing recall, or minimizing false positives).
