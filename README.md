# 📡 Telecom CDR Analytics Platform

**Scalable Clustering of Heterogeneous Big Data via Distributed Ensemble Gradient-Boosting Models  
(XGBoost on Spark/Dask), Applied to Telecom Network Call Detail Record (CDR) Analysis**

---

## 🎯 Project Objective

This project provides an end-to-end Big Data & Machine Learning analytics engine for telecommunication networks. It processes heterogeneous Call Detail Records (CDRs), discovers behavioural subscriber segments via unsupervised clustering, detects anomalous calling activities, and classifies fraud using an ensemble of distributed gradient-boosted decision trees (XGBoost) accelerated by Dask and Apache Spark.

---

## 🗂️ File Structure

```
C:/Telecom-call/
├── config.py               ← Central configuration (paths, dataset sizes, ML hyperparams)
├── data_generator.py       ← Synthetic heterogeneous CDR dataset generator (Faker + NumPy)
├── preprocessing.py        ← Dask parallel processing pipeline + subscriber feature engineering
├── clustering.py           ← K-Means clustering + Elbow/Silhouette analysis + Dynamic segment naming
├── anomaly_detection.py    ← Unsupervised Isolation Forest anomaly detection
├── xgb_classifier.py       ← Distributed histogram-based XGBoost fraud classifier (incl. cluster features)
├── spark_pipeline.py       ← Enterprise multi-node PySpark + SparkXGBClassifier pipeline
├── benchmark.py            ← Cross-size performance benchmarking (10K, 50K, 200K)
├── run_pipeline.py         ← End-to-end automated CLI pipeline execution
├── dashboard.py            ← Client-facing Streamlit dashboard (4 focused modules)
├── requirements.txt        ← Python dependencies
├── data/                   ← Generated CSV/Parquet datasets
├── models/                 ← Serialized ML model artefacts (.pkl)
└── reports/                ← Visualisations, ROC curves, feature importances, and benchmarks
```

---

## ⚙️ Setup & Quick Execution

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the Full End-to-End Pipeline (1-Click)

```bash
python run_pipeline.py --size small
```

This automatically executes:
1. **Data Generation** (10,000 synthetic heterogeneous CDR records)
2. **Dask Parallel Preprocessing** (Cleaning, temporal feature engineering, and per-subscriber aggregation)
3. **K-Means Clustering & Profiling** (Dynamic segment labeling: Heavy Callers, International Travellers, Night-Shift, High-Risk)
4. **Isolation Forest Anomaly Detection** (Unsupervised oddity scoring)
5. **XGBoost Fraud Classification** (Supervised gradient-boosting with cluster tags, 5-fold cross-validation)
6. **Performance Benchmarking** (Wall-clock timing & metrics)

### 3. Launch the Clean Web Dashboard

```bash
streamlit run dashboard.py
```
Open **http://localhost:8501** in your browser.

---

## 📊 Dashboard Modules (Client Perspective)

| Module | Purpose & Insights |
|---|---|
| **📊 Overview** | Executive summary KPIs (Calls, Subscribers, Revenue, Fraud rate), Daily Call Trend, Call Mix, Model Metrics, and Title Architecture mapping. |
| **📞 Call Analytics** | Network usage analytics, Peak vs Off-peak hourly bar chart, Duration histograms, Weekly Day-Hour Heatmap, Revenue per Call Type. |
| **🚨 Fraud Detection** | Confirmed Fraud table, AI-flagged Unsupervised Anomalies, and side-by-side behavioral comparison bars (International %, Night %, Charges). |
| **👥 User Segments** | 5 dynamically categorized subscriber cards, Subscriber breakdown per segment, Fraud risk per segment, and 2-D PCA behavioural space map. |

---

## 🏛️ Architecture & Title Alignment

| Title Term | Implementation |
|---|---|
| **Scalable** | Validated across `small` (10,000), `medium` (50,000), and `large` (200,000) records with automated timing benchmarks. |
| **Clustering** | Unsupervised K-Means clustering with Elbow (WCSS) & Silhouette scoring, mapped into 2D PCA space with dynamic behavioral naming. |
| **Heterogeneous Big Data** | Ingests mixed CDR fields: numerical (durations, charges), timestamps (call start), categoricals (call type, direction), binary flags, and geospatial tower coordinates. |
| **Distributed (Dask & Spark)** | Parallel partitioned execution using **Dask** for local multi-core environments, plus a standalone **PySpark** cluster module (`spark_pipeline.py`). |
| **Ensemble Gradient-Boosting** | Histogram-based tree ensemble (`tree_method='hist'`) in **XGBoost**, utilizing subscriber metrics and cluster memberships to predict fraud with >99% ROC-AUC. |
| **Telecom CDR Analysis** | Provides actionable business intelligence: network load management, billing analysis, and fraud mitigation. |

---

## 🎓 Viva & Technical Q&A Reference

1. **Why use Dask for data preprocessing?**  
   Dask reads CSV data in parallel partitions, avoiding out-of-memory errors and utilizing all available CPU cores natively in Python without requiring a dedicated Java/Hadoop cluster.

2. **How does Clustering integrate with Gradient-Boosting?**  
   The unsupervised K-Means clusters identify behavioral archetypes. These cluster IDs are fed directly as structural features into XGBoost, allowing the gradient-boosted trees to partition decisions based on subscriber profiles.

3. **How does XGBoost handle class imbalance?**  
   In telecommunications, fraud is rare (~5%). We compute `scale_pos_weight = (1 - fraud_fraction) / fraud_fraction` (~19) to penalize false negatives on the minority fraud class.

4. **Why combine Isolation Forest with XGBoost?**  
   XGBoost is supervised (requires known labels). Isolation Forest is unsupervised (requires no labels). In production, Isolation Forest flags novel, unseen zero-day fraud patterns which are later reviewed and fed to XGBoost.

---

## 📦 Tech Stack

- **Data Processing:** Python 3.14, Dask, Pandas, NumPy, PyArrow
- **Machine Learning:** XGBoost, Scikit-Learn
- **Distributed Computing:** Dask DataFrame, PySpark (`spark_pipeline.py`)
- **Dashboard & Visualization:** Streamlit, Plotly Express & Graph Objects, Matplotlib, Seaborn
