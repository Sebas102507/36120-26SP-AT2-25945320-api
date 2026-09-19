# models/

Production artefacts loaded by the API at start-up. They are produced in the experiments
repository and copied here with `make artefacts`, or automatically by the experiments
repository's `sync-production-models.yml` workflow.

```text
models/
├── preprocessing/
│   └── feature_pipeline.joblib     fitted nucleus_ml preprocessing pipeline (hourly -> feature row)
├── comfort_climate/                 promoted Climate Comfort Index bundle
│   ├── <champion>_h1.joblib          model for +1 day
│   ├── <champion>_h2.joblib          model for +2 days
│   ├── <champion>_h3.joblib          model for +3 days
│   ├── manifest.json                 model files and each model's feature columns
│   ├── metadata.json                 served by /model-metadata
│   ├── drift_reference.json          training feature distribution, used by /monitoring/drift
│   └── promotion.json                audit record of how the champion was chosen
└── weather_hazard/                  promoted Weather Hazard Category bundle
    ├── <champion>_h7.joblib          model for +7 days
    └── manifest.json, metadata.json, drift_reference.json, promotion.json
```

The API reads this folder from `MODEL_DIR`: `./models` when run locally, `/models` inside the
Docker image. The `.joblib` files are pickled objects; load them only from this trusted source.
