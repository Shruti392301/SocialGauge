# SocialPulse-Analytics 🚀

*Data-Driven Social Engagement & Predictive Virality Architecture*[cite: 1]

[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/Streamlit-1.28%2B-FF4B4B.svg)](https://streamlit.io/)
[![BI Tool](https://img.shields.io/badge/Power%20BI-Desktop-yellow.svg)](https://powerbi.microsoft.com/)

SocialPulse-Analytics is an end-to-end data pipeline, machine learning engine, and interactive dashboard built to analyze multi-platform social performance, predict content virality, and extract deep linguistic audience insights[cite: 1].

---

## 📌 Table of Contents
- [Key Capabilities](#-key-capabilities)
- [Tech Stack](#-tech-stack)
- [Dataset Architecture](#-dataset-architecture)
- [Key Project Findings](#-key-project-findings)
- [Pipeline Architecture](#-pipeline-architecture)
- [Quickstart](#-quickstart)

---

## 🔥 Key Capabilities

* **Multi-Channel Pipeline:** Processes 5,000 post records across TikTok, Instagram, Twitter, and YouTube with standardized schema normalization[cite: 1].
* **Predictive Virality Engine:** Uses Random Forest Regressors and Classifiers to model share-driven viral coefficients and score content reach potential[cite: 1].
* **NLP & Sentiment Processing:** Analyzes 37,249 comment records using NLTK and TextBlob for polarity scoring and problem-intent trigger detection[cite: 1].
* **Statistical A/B Framework:** Performs Welch’s two-sample t-tests to evaluate significance in engagement rates across formats (Shorts, Reels, Posts, Live Streams)[cite: 1].
* **Optimization Recommender:** Employs a multi-factor composite scoring engine to rank format-topic combinations and maximize campaign ROI[cite: 1].
* **Time-Series Forecasting:** Evaluates hashtag trajectory velocity ($y = mx + c$) to automatically classify emerging, stable, and declining trends[cite: 1].

---

## 🛠 Tech Stack

* **Language:** Python 3.10+
* **Data Processing & ML:** Pandas, NumPy, Scikit-Learn, SciPy, NLTK, TextBlob[cite: 1]
* **Dashboards & UI:** Streamlit (`app.py`), Power BI Desktop[cite: 1]

---

## 📊 Dataset Architecture

| Dataset Component | Total Records | Key Metrics / Features |
| :--- | :--- | :--- |
| **Primary Content**[cite: 1] | 5,000 Posts[cite: 1] | Views, Likes, Shares, Comments, Engagement %[cite: 1] |
| **Audience Corpus**[cite: 1] | 37,249 Comments[cite: 1] | Sentiment Polarity, Subjectivity, Keyword Triggers[cite: 1] |

---

## 📈 Key Project Findings

* **Format Winner:** Shorts deliver a **1.6x engagement multiplier** (80.63% average engagement) over standard formats[cite: 1].
* **Top Topic:** `#Viral` achieved the highest average viral coefficient (0.2296) and total shares[cite: 1].
* **Audience Intent:** 8.05% of comments explicitly highlight problem-awareness triggers (e.g., *help*, *issue*, *problem*) ideal for conversion targeting[cite: 1].

---

## 🏗 Pipeline Architecture

```text
📁 SocialPulse-Analytics
 ├── 📄 app.py                  # Streamlit Dashboard App
 ├── 📄 SocialPulse_Pipeline.ipynb # Jupyter Notebook Data & ML Pipeline
 ├── 📄 Cleaned_Viral_Social_Media_Trends.csv # Cleaned Primary Post Dataset
 ├── 📄 Social media_Data.csv   # Secondary Sentiment Comment Dataset
 ├── 📄 requirements.txt        # Python Dependencies
 └── 📄 README.md               # Repository Documentation

```
## 🚀Quickstart
Clone the repository:

Bash
git clone [https://github.com/your-username/SocialPulse-Analytics.git](https://github.com/your-username/SocialPulse-Analytics.git)
cd SocialPulse-Analytics
Install dependencies:

Bash
pip install -r requirements.txt
Launch the interactive Streamlit app:

Bash
streamlit run app.py
