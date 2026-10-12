# Optimizing ROI on Digital Marketing Campaigns – AeroReach Insights

**MBA Semester IV Research Project · Jain (Deemed-to-be University), Bengaluru**
A data-driven study that predicts which users are likely to buy an airline ticket and segments them by behaviour, so digital marketing spend can go where it earns the most.

**Tools:** Python (Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn) · K-Means clustering · Logistic Regression, Random Forest, Gradient Boosting

---

## 1. Business Problem

Airlines spend heavily on digital marketing but cannot tell which engagement behaviours actually lead to ticket purchases, so budgets are spread across broad audiences. This project asks:

> Which customer engagement behaviours are most strongly associated with ticket purchase, and can a model identify likely buyers early enough to guide campaign targeting?

### Objectives
1. Run a full exploratory data analysis (EDA) of the engagement data.
2. Clean the data (missing values, inconsistent labels, outliers).
3. Engineer business-friendly engagement scores.
4. Segment users with K-Means clustering.
5. Build and compare predictive models for purchase.
6. Turn the findings into targeting and budget recommendations.

---

## 2. Dataset

| Item | Detail |
|---|---|
| File | `AeroReach Insights.csv` (plus `Data Description.docx`) |
| Size | 11,760 users × 17 columns |
| Target | `Taken_product` (Yes/No): whether the user bought a ticket |
| Class balance | 16.1% Yes, 83.9% No (imbalanced) |

**Key columns:** Yearly_avg_view_on_travel_page, Daily_Avg_mins_spend_on_traveling_page, Yearly_avg_comment_on_travel_page, total_likes_on_outstation_checkin_given / received, yearly_avg_Outstation_checkins, week_since_last_outstation_checkin, following_company_page, montly_avg_comment_on_company_page, travelling_network_rating, preferred_device, preferred_location_type, member_in_family, working_flag.

**Data quality issues found:** missing values in 7 columns, inconsistent labels (e.g. "ANDROID" / "Android OS" / "Android"; stray "1"/"0" in a Yes/No column; "Three" typed as text), a numeric column stored as text, and a corrupted `Adult_flag(Age Group)` column.

---

## 3. Project Steps

### Step 1 – Define the problem (Synopsis)
Set the research title, problem statement, objectives and planned methods.

### Step 2 – Understand the data
Loaded the CSV, checked shape, data types, missing values and the target balance.

### Step 3 – Exploratory data analysis
- **Univariate:** distributions of time on page, views, likes, comments and recency.
- **Bivariate:** each engagement variable against purchase outcome, plus purchase rate by device, company-page follow and location type.
- **Correlation heatmap** across numeric features.

### Step 4 – Clean and preprocess
| Issue | Treatment |
|---|---|
| Device label variants | Standardised to Android, iOS, iOS and Android, Tablet, Laptop, Mobile, Other |
| Stray "1"/"0" in company-page follow | Recoded to Yes/No |
| "Three" in family members | Converted to 3 (numeric) |
| Outstation check-ins stored as text | Converted to numeric |
| Missing numeric values | Median imputation |
| Missing categorical values | Mode imputation |
| Outliers | Capped using the IQR method (values kept, not removed) |
| Corrupted `Adult_flag(Age Group)` (~94% of rows) | Dropped |

### Step 5 – Feature engineering
Four composite scores, each built from min-max scaled features:

| Score | Built from |
|---|---|
| Engagement score | Daily minutes, yearly views, yearly comments on travel pages |
| Social influence score | Likes given, likes received, travel network rating (inverted) |
| Recency score | Inverse of weeks since last outstation check-in |
| Brand engagement score | Company page follow + monthly comments on company page |

### Step 6 – Customer segmentation (K-Means)
Applied K-Means on the four scores. The elbow method gave **k = 3**.

| Cluster | Segment | Purchase rate | Users |
|---|---|---|---|
| 0 | High-Value Converters (high recency and brand engagement) | ~27% | 3,236 |
| 1 | Moderately Engaged Observers | ~16% | 2,641 |
| 2 | Passive / Low-Potential Users | ~10% | 5,883 |

### Step 7 – Handle class imbalance
Split the data 80/20 (stratified), scaled the features, and oversampled the minority ("Yes") class in the training set with a custom SMOTE-style k-nearest-neighbour method, leaving the test set untouched.

### Step 8 – Model building and validation
Trained three models on seven features (the four scores plus daily minutes, yearly comments and travel network rating) and evaluated them on the held-out test set of 2,352 users:

| Model | Accuracy | Precision (Yes) | Recall (Yes) | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | ~67% | ~28% | ~68% | ~0.40 | ~0.73 |
| Random Forest | ~82% | ~45% | ~61% | ~0.52 | ~0.84 |
| **Gradient Boosting (final)** | **~90%** | **~71–72%** | **~63%** | **~0.67** | **~0.92** |

The final model correctly identified about 63% of actual buyers while flagging only about 5% of non-buyers as likely buyers.

### Step 9 – Interpret the model
Permutation feature importance shows the most influential features are the **social influence score** and **brand engagement score**, followed by travel network rating and recency. Raw browsing volume (daily minutes, yearly comments) ranks lowest.

### Step 10 – Business recommendations
See section 5.

---

## 4. Key Findings

- **Only 16.1% of users buy**, so accuracy alone is misleading; precision, recall and ROC-AUC were used.
- **Browsing volume does not predict buying.** Purchasers did not spend more time or views than non-purchasers; the correlation with the target was weak (the strongest was about -0.17).
- **Brand-page followers convert about 3× more** (~30% vs ~10%).
- **Recency and brand engagement define the best segment**, not raw engagement: the segment with the highest engagement score converts at a lower rate than the High-Value Converters.
- **A non-linear model is needed:** Logistic Regression performs poorly, while Gradient Boosting reaches ROC-AUC of about 0.92.
- The interim report's hypothesis that "more engagement means more purchases" was revised by the final analysis.

---

## 5. Recommendations

1. **Retarget within 1–4 weeks** of a user's last outstation check-in.
2. **Invest in brand-page engagement** (polls, Q&A, loyalty programmes) as a conversion lever.
3. **Put the largest budget on High-Value Converters**, nurture Moderately Engaged Observers, and keep direct spend on Passive users minimal.
4. **Use the model as a scoring layer** to flag high-probability buyers for personalised campaigns.
5. **Track depth metrics** (comments, follows, recency) rather than page views.

**Limitations and next steps:** no campaign cost data (so no CPA or ROAS yet), a corrupted age column, and recall of about 63% that could improve with threshold tuning. Future work includes platform-wise analysis, comment text analytics, real-time scoring and A/B testing.

---

## 6. How to Run

1. Place `AeroReach Insights.csv` in the working directory (update the file name in the script if yours differs).
2. Install the libraries: `pip install pandas numpy scikit-learn matplotlib seaborn`
3. Run the scripts in order: data cleaning and feature engineering, K-Means clustering, model training and evaluation, then the charts script.

## 7. Repository Structure

```
├── AeroReach Insights.csv            # Dataset
├── Data Description.docx             # Column descriptions
├── aeroreach_pipeline.py             # Cleaning, features, clustering, models (from the code PDF)
├── Synopsis.pdf                      # Research synopsis
├── Interim_Report.pdf                # Interim report
├── Final_Report.pdf                  # Final report
└── README.md
```

## 8. Skills Demonstrated

Data cleaning · EDA · Feature engineering · K-Means segmentation · Class-imbalance handling · Classification modelling and validation · Business storytelling

---

**Author:** Astha Kumari Jha, MBA (Data Science & Analytics), Jain University
**Guide:** Hrushikesha Shastry
**GitHub:** [github.com/Asthajha4275](https://github.com/Asthajha4275)
