# Customer Segmentation — Online Retail II

Unsupervised machine learning project for customer segmentation using clustering on real e-commerce transaction data.

## Overview

Applied K-Means and DBSCAN to segment ~4,300 customers of a UK-based online retailer into actionable profiles for marketing strategy. The project covers the full data science pipeline: data cleaning, feature engineering, model selection, and business interpretation.

**Dataset:** [Online Retail II — UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/502/online+retail+ii)  
~500k transactions from 2010–2011.

---

## Pipeline

```
Raw data (541k rows)
    → Cleaning (nulls, cancellations, negative prices/quantities)
    → RFM+ feature engineering (1 row per customer)
    → Winsorization (1–99%) + Log transform + RobustScaler
    → Clustering (K-Means / DBSCAN)
    → Business interpretation
```

---

## Features (RFM+)

| Feature | Description |
|---|---|
| `DIAS_ULTIMA_COMPRA` | Days since last purchase (Recency) |
| `CANT_FACTURAS` | Number of distinct invoices (Frequency) |
| `GASTO_TOTAL` | Total spend in GBP (Monetary) |
| `TICKET_PROMEDIO` | Average spend per invoice |
| `PRODUCTOS_DISTINTOS` | Number of distinct products purchased |

---

## Models

### K-Means (k=4) — Final model
Selected via elbow method + silhouette score. Silhouette: **0.2511**

| Segment | Customers | Recency | Frequency | Total Spend |
|---|---|---|---|---|
| Premium | 984 | 22 days | 11 invoices | £6,667 |
| Por Desarrollar | 1,120 | 36 days | 4 invoices | £817 |
| Ocasionales de Alto Ticket | 1,187 | 143 days | 2 invoices | £1,014 |
| Perdidos | 1,047 | 158 days | 1 invoice | £199 |

### DBSCAN (eps=0.7, min_samples=5)
Parameters selected via grid search balancing silhouette, cluster count and noise ratio. Silhouette: **0.3419**

| Result | Count |
|---|---|
| Cluster 0 — Standard clients | 4,297 |
| Cluster 1 — Mega wholesalers | 11 |
| Noise | 30 (0.7%) |

**Key finding:** DBSCAN identified 11 mega wholesalers (avg spend £120,727) that K-Means had absorbed into the Premium segment, diluting their real profile.

---

## Model Selection

Despite DBSCAN achieving a higher silhouette score, **K-Means was selected as the final model** because DBSCAN's 2-cluster structure (standard clients vs. mega wholesalers) is too coarse for differentiated marketing strategies. K-Means delivers 4 balanced, interpretable segments with direct business applicability.

The two models are complementary: K-Means for segmentation, DBSCAN for outlier detection.

---

## Key Decisions

- **Why not cluster on raw variables?** Clustering directly on transaction-level variables produces degenerate results (4,335 of 4,338 customers in a single cluster, silhouette artificially inflated to 0.99). RFM aggregation is necessary.
- **Why log transform + winsorization?** RFM variables are heavily right-skewed. Winsorization caps extreme values; log transform corrects residual skew. Both steps are complementary.
- **Why RobustScaler?** Uses median and IQR, making it robust to outliers that survive preprocessing.
- **Why not PCA?** With only 5 well-defined features, dimensionality reduction would sacrifice interpretability without meaningful benefit.

---

## Limitations

- Customers without `CustomerID` (~25% of rows) were excluded — no profile can be built without an identifier.
- Cancelled invoices (prefix `C`) were removed, but matching them to their original invoices proved unreliable: only 3.6% of cancellations matched by CustomerID + StockCode + Quantity + date. Simple removal was retained and documented.
- Some customer profiles may have overestimated spend if their original invoice was cancelled but not matched.

---

## Stack

- Python 3 / Jupyter Notebook
- pandas, numpy, scikit-learn
- matplotlib, seaborn, plotly

