## Text Feature Engineering with Azure ML

This repository contains the Azure Machine Learning feature engineering pipeline for transforming Amazon Electronics review text into machine-learning features. The pipeline computes text-based features and registers them in the Azure ML Feature Store for reuse in downstream modeling.

Codes on the other branch.

#### Project Structure
```
├── components/
│   ├── split_dataset/        # Train / validation / test split
│   ├── normalize_text/       # Text normalization
│   ├── length_features/      # Review length features
│   ├── sentiment_features/   # Sentiment analysis (VADER / TextBlob)
│   ├── tfidf_features/       # TF-IDF features (train-only fit)
│   ├── sbert_embeddings/     # Sentence-BERT embeddings
│   └── merge_features/       # Merge all feature outputs
│
├── pipelines/
│   └── feature_pipeline.yml  # Azure ML feature engineering pipeline
│
├── feature_store/
│   ├── entity_amazon_review.yml
│   ├── FeatureSetSpec.yaml
│   └── feature_set_amazon_review_text_features.yml
│
├── data/
│   └── features_v1_sampled.yml   # Sampled Gold dataset definition
│
├── datastores/
│   └── curated_adls.yml           # Azure Data Lake datastore (keys excluded)
│
└── README.md
```

#### Features Generated
- Review length (characters, words)
- Sentiment scores (positive, negative, neutral, compound)
- TF-IDF vectors (fit on training split only)
- Sentence-BERT semantic embeddings

#### Pipeline Overview
1. Load sampled Gold dataset
2. Split data (train / validation / test) to prevent leakage
3. Normalize review text
4. Extract text-based features
5. Merge all features into a single dataset
6. Register features in the Azure ML Feature Store
