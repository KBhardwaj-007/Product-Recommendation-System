# Product Recommendation System : 🎮 Video Games

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/KBhardwaj-007/Product-Recommendation-System/blob/main/Product_Recommendation_System.ipynb)

A comprehensive, end-to-end machine learning project that deploys a **SVD-Powered User-to-Item Personalized Hybrid Recommender** for video games. It predicts explicit user ratings using advanced matrix factorization, moving beyond simple item similarity to deliver unparalleled personalization. The entire pipeline, from MongoDB data ingestion to a production-ready Streamlit web app, is implemented for maximum efficiency and real-world impact.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Project Architecture](#project-architecture)
- [Dataset](#dataset)
- [Methodology](#methodology)
- [Results & Insights](#results--insights)
- [Installation](#installation)
- [Usage](#usage)
- [Visualizations](#visualizations)
- [Technologies Used](#technologies-used)
- [Business Recommendations](#business-recommendations)
- [Future Enhancements](#future-enhancements)

---

## 🎯 Overview

In the highly competitive e-commerce landscape, personalized recommendations are the core engine for driving conversions. This project solves the personalization challenge by developing a sophisticated **Dual-Hybrid Recommendation System** for video games using 230,000+ Amazon reviews. The system's flagship feature is the **User-to-Item Personalized Recommender**, which uses the highly accurate SVD algorithm to predict exactly what a specific user will love.

### Key Objectives

1. **Build a Personalized User-to-Item Model** using the optimal SVD Matrix Factorization algorithm.
2. Develop a **Dual-Hybrid Model** combining SVD prediction, content-based features, and popularity for robust suggestions.
3. Establish a fast data pipeline from MongoDB to a trained, serialized model (`.joblib`) for near-instantaneous inference.
4. Perform accurate sentiment analysis on review text using ML classifiers (LightGBM, XGBoost).
5. **Deploy an interactive Streamlit application** showcasing both Personalized and Item-to-Item results.

---

## ✨ Key Features

- **SVD-Powered Personalized Recommender**: Predicts explicit ratings for unrated products based on individual user latent factors.
- **Dual-Hybrid Engine**: Offers two modes: **Personalized** (User-to-Item) for engagement and **Item-to-Item** for product similarity.
- **Matrix Factorization Optimality**: SVD demonstrated superior accuracy (RMSE: 1.0823) with fast training time (Avg. 2.72s).
- **Sentiment Analysis**: Classifies review sentiment with high-accuracy models (LightGBM F1-Score $\approx$ 0.90).
- **Fast Inference**: All matrices and the SVD model are pre-computed and saved for near-real-time performance in the web application.
- **Interactive Web App (3 Pages)**: Streamlit-based UI for real-time recommendation generation.
- **Modular Architecture**: Clean separation of concerns (`mongo_connection`, `hybrid_personalized`, etc.) for scalability.

---

## 🏗️ Project Architecture

```
Product_Recommendation/
│
├── data/
│   ├── video_games_reviews.csv          # Raw dataset
│   ├── cleaned_reviews.joblib           # Processed data
│   ├── svd_model.joblib                 # 🌟 TRAINED SVD USER-TO-ITEM MODEL
│   ├── all_products.joblib              # List of all ASINs (for SVD prediction)
│   ├── cf_sim_df.joblib                 # Item-to-Item CF similarity matrix
│   ├── tfidf_matrix.joblib              # Content-based TF-IDF matrix
│   └── ml_results.joblib                # ML model results & metrics
│
├── src/
│   ├── logger_config.py
│   ├── mongo_connection.py
│   ├── data_preprocessing.py
│   ├── baseline.py
│   ├── collaborative.py
│   ├── content_based.py
│   ├── hybrid.py
│   ├── hybrid_fast.py                   # Item-to-Item Hybrid Logic
│   └── hybrid_personalized.py           # 🌟 USER-TO-ITEM HYBRID LOGIC
│   └── ml_models.py
│
├── app/
│   └── streamlit_app.py                 # Web application (3 Pages)
│
├── notebooks/
│   └── Product_Recommendation_System.ipynb  # Main notebook
│
└── README.md
```

---

## 📊 Dataset

**Source**: Amazon Video Game Reviews  
**Size**: 231,780 entries  
**Time Period**: 2000-2014

### Features

| Column | Description |
|--------|-------------|
| `reviewerID` | Unique identifier for the reviewer |
| `asin` | Unique product identifier |
| `reviewerName` | Display name of the reviewer |
| `helpful` | Helpfulness votes [helpful_votes, total_votes] |
| `reviewText` | Full review text |
| `overall` | Star rating (1-5) |
| `summary` | Review title/summary |
| `unixReviewTime` | Unix timestamp |
| `reviewTime` | Readable date format |

### Engineered Features

- `helpful_ratio`: Proportion of helpful votes
- `helpful_votes`: Total helpful votes received
- `label`: Binary sentiment (1=positive: $\ge 4$, 0=negative: $<4$)
- `reviewTime`: Standardized datetime format

---

## 🔬 Methodology

### 1. Data Pipeline

```python
Raw Data → MongoDB → Preprocessing → Feature Engineering → Model Training → Serialization (.joblib) → Deployment
```

**Key Steps**: Data cleaning, parsing `helpful` votes, and creating a binary sentiment target `label`.

### 2. Recommendation Models (Core Logic)

#### Personalized User-to-Item Hybrid
- **Model:** $\text{SVD Prediction} \times \mathbf{\alpha} + \text{Popularity Score} \times \mathbf{\beta} + \text{Content Score} \times \mathbf{\gamma}$
- **CF Core:** **SVD Matrix Factorization** predicts the user's rating for unrated items.
- **Content-Based:** Item similarity calculated based on the user's **highest-rated game**.
- **Weights:** Optimized as $\mathbf{\alpha=0.5}$ (SVD Prediction), $\mathbf{\beta=0.3}$ (Popularity), $\mathbf{\gamma=0.2}$ (Content).

#### Benchmarking & Item-to-Item Models
- **SVD, BaselineOnly, NMF**: Evaluated using **RMSE** and **MAE** to select the optimal algorithm for the personalized model.
- **Item-to-Item Hybrid**: A faster fallback model using item similarity on pre-computed matrices.

### 3. Sentiment Analysis

**Models Trained**: RandomForest, XGBoost, LightGBM (using TF-IDF on `reviewText`)
**Evaluation Metrics**: Accuracy, Precision, Recall, F1-Score, ROC AUC
**Goal**: Validate review sentiment to provide granular market intelligence alongside recommendations.

---

## 📈 Results & Insights

### Collaborative Filtering Performance (Matrix Factorization)

| Algorithm | Mean RMSE (Error) | Mean MAE (Error) | Mean Fit Time |
|-----------|-------------------|------------------|---------------|
| **SVD** | **1.0823** | **0.8344** | $\\approx$ **2.72s** |
| **BaselineOnly** | 1.0875 | 0.8473 | $\\approx$ 1.10s |
| **NMF** | 1.2749 | 0.9809 | $\\approx$ 6.64s |

**Conclusion**: **SVD** is the optimal model, providing the lowest predictive error with superior efficiency compared to NMF.

### Sentiment Classification Performance

| Model | Accuracy | Precision | Recall | F1-Score | ROC AUC |
|-------|----------|-----------|--------|----------|---------|
| **LightGBM** | **0.840** | **0.851** | 0.955 | **0.900** | **0.874** |
| **XGBoost** | 0.838 | 0.847 | 0.959 | 0.899 | 0.872 |
| **RandomForest** | 0.822 | 0.822 | **0.977** | 0.892 | 0.860 |

**Conclusion**: LightGBM is the top-performing classifier. The high **Recall** for the positive class ($\approx$ 95.5%) suggests the models are excellent at identifying positive reviews but struggle slightly more with the less frequent negative class (due to data imbalance).

---

## 🚀 Installation

### Prerequisites

```bash
Python 3.8+
MongoDB (local or cloud instance)
Google Colab (recommended) or Jupyter Notebook
```

### Setup

1. **Clone the repository**
```bash
git clone https://github.com/KBhardwaj-007/Product-Recommendation-System.git
cd Product-Recommendation-System
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```
Required packages:
```
pandas
numpy
scikit-learn
matplotlib
seaborn
nltk
pymongo
wordcloud
surprise
xgboost
lightgbm
streamlit
joblib
pyngrok
```

3. **Configure secrets** (in Google Colab)

Add to Colab Secrets:
- `MONGO_URI`: Your MongoDB connection string
- `NGROK_TOKEN`: Your ngrok authentication token

4. **Download dataset**

Place `video_games_reviews.csv` in the `data/` directory.

---

## 💻 Usage

### Running the Notebook

Open `Product_Recommendation_System.ipynb` in Google Colab and run all cells. The pipeline automatically:
1.  Loads data into MongoDB.
2.  Preprocesses data.
3.  Trains and serializes the final **SVD model** and all necessary matrices.
4.  Generates and displays comparison results for all models.

### Launching the Web App

The final notebook cells deploy the application via Streamlit and expose it via a public ngrok URL.

```python
# In the notebook, execute:
!streamlit run app/streamlit_app.py &>/dev/null &

# Create public tunnel
from pyngrok import ngrok
public_url = ngrok.connect(addr="8501")
print(f"🎉 App live at: {public_url}")
```

### Using the Streamlit App (3 Pages)

1.  **👤 Personalized Recommender**: Select a **User ID** to receive a ranked list of games predicted to be rated $\mathbf{\ge 4.5}$ stars by that specific user.
2.  **🔗 Item-to-Item Recommender**: Select a **Product ID** to find similar games based on combined user behavior and review content.
3.  **📊 Model Performance**: View and analyze all classification and collaborative filtering results, heatmaps, and the Confusion Matrix.

---

## 📊 Visualizations

### Distribution of Ratings
![Distribution of Overall Ratings](plots/distribution_of_overall_ratings.png)

*Analysis*: Strong positive skew with 58% 5-star and 27% 4-star reviews, indicating high customer satisfaction.

---

### Helpfulness Ratio Distribution
![Helpfulness Ratio](plots/distribution_of_helpfulness_ratio.png)

*Analysis*: Bimodal distribution with peaks at 0.0 (unvoted) and 1.0 (unanimously helpful), suggesting polarized community engagement.

---

### Reviews Over Time
![Reviews Per Month](plots/number_of_reviews_per_month.png)

*Analysis*: Exponential growth from 2012-2014, peaking at 6,000+ monthly reviews, with strong seasonal patterns.

---

### Top Products & Reviewers
![Top 10 Analysis](plots/10_most_reviewed_products.png)

*Analysis*: High concentration with top product receiving 800 reviews and most active reviewer contributing 780 reviews.

---

### Review Summary Word Cloud
![Word Cloud](plots/most_common_words.png)

*Analysis*: Dominant positive terms ("Great", "Good", "Best", "Awesome") with gaming-specific vocabulary ("Game", "Play", "PS3").

---

### Sentiment Model Comparison
![Model Performance](plots/comparision_of_sentiment_classification_models.png)

*Analysis*: LightGBM leads with 84.3% accuracy; all models show high recall (>95%) but lower precision due to class imbalance.

---

### Collaborative Filtering Comparison
![CF Algorithms](plots/comparision_of_surprise_cf_algorithms.png)

*Analysis*: SVD achieves lowest error rates (RMSE: 1.09); BaselineOnly offers best speed-accuracy tradeoff.

---

### Confusion Matrix
![Confusion Matrix](plots/confusion_matrix_for_lightgbm.png)

*Analysis*: LightGBM correctly classifies 33,422 positive reviews but generates 5,832 false positives due to 3:1 class imbalance.

---

## 🛠️ Technologies Used

| Category | Technologies |
|----------|-------------|
| **Languages** | Python 3.8+ |
| **Data Processing** | Pandas, NumPy |
| **Machine Learning** | Scikit-learn, XGBoost, LightGBM, Surprise |
| **NLP** | NLTK, TF-IDF Vectorizer |
| **Database** | MongoDB, PyMongo |
| **Visualization** | Matplotlib, Seaborn, WordCloud |
| **Web Framework** | Streamlit |
| **Deployment** | ngrok, Google Colab |
| **Utilities** | Joblib, tqdm |

---

## 💼 Business Recommendations

### 1. 🥇 Maximize Sales with SVD Personalization

**Action**: **Immediately deploy the SVD-Powered Personalized Hybrid Model (User-to-Item)** for real-time inference on the homepage, checkout, and email campaigns.

**Impact**: Maximize revenue by showing each user the few items they are **most likely to purchase** (based on predicted high rating), leading to conversion rates significantly higher than generic top-seller lists.

### 2. 🛡️ Implement a Dynamic Cold-Start Strategy

**Action**: Use a conditional system:
- **New Users ($\le 1$ review)**: Default to the `weighted_popularity_based` model.
- **New Items (No reviews)**: Use the **Content-Based** module based on product description/metadata.
- **Active Users**: Use the SVD-powered Personalized Hybrid.

**Impact**: Guarantees a relevant recommendation experience from the first interaction, retaining new users who lack history.

### 3. 💸 Trigger High-Confidence Bundling

**Action**: Use the SVD prediction score as a campaign trigger. If a user's predicted rating for a new or high-margin game is **$\mathbf{\ge 4.5}$**, automatically create and send a targeted bundle offer.

**Impact**: Converts high-confidence intent into higher-value sales, improving Average Order Value (AOV).

### 4. 📉 Real-Time Sentiment & Inventory

**Action**: Apply the trained LightGBM model to incoming reviews in real-time. Create an alert system for any product whose **Negative sentiment exceeds a 20% threshold** for quick review and potential inventory adjustment.

**Impact**: Provides early warning for product issues, mitigating financial risk and protecting brand reputation.

---

## 🔮 Future Enhancements

- [ ] **Real-Time Retraining Pipeline**: Automate the SVD model re-training nightly using new data on a scalable cloud resource (e.g., AWS Lambda/GCP Cloud Functions).
- [ ] **A/B Test Integration**: Build a logging framework to compare conversion rates between the old Item-to-Item and the new Personalized Hybrid model in a live environment.
- [ ] **Multi-Modal Features**: Integrate game metadata (e.g., Genre, Developer, Release Year) into the SVD feature matrix for deeper latent factor modeling.
- [ ] **Mobile Optimization**: Deploy a lighter-weight, mobile-friendly Streamlit interface.

<p align="center">
  <strong>⭐ If you found this project useful, please consider giving it a star! ⭐</strong>
</p>

<p align="center">
  © 2025 Product Recommendation System | Powered by Python 🐍
</p>
