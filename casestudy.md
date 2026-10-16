# SwiftDrop Logistics: System Audit & Strategy Report

## 1. ML Problem Definition

### Problem Type: Regression

SwiftDrop's delivery-time prediction problem is a **Regression task** because the model must predict a continuous numerical value: the estimated time required to deliver a package, measured in minutes. It is not a classification problem because the primary objective is to predict an exact delivery duration rather than assign a discrete category.

### Mathematical Objective

The model should predict the estimated delivery time in minutes using information available at the moment a delivery prediction is requested.

**Primary optimization metric: Mean Absolute Error (MAE).**

MAE measures the average absolute difference between predicted delivery times and actual delivery times.

For example, if the model predicts 15 minutes but the actual delivery takes 45 minutes, the absolute error is 30 minutes.

The objective is to minimize MAE on unseen delivery data, reducing the gap between promised and actual delivery times.

MAE is appropriate because it is expressed in minutes, is easy for business stakeholders to interpret, and is less sensitive to extreme errors than Mean Squared Error (MSE).

### Ground Truth

The ground truth will be the **actual elapsed delivery time**, measured in minutes, from the confirmed start of the delivery to the confirmed completion of the delivery.

These timestamps must come from reliable operational records, such as courier pickup and package-delivered events. Cancelled deliveries, duplicate events, invalid timestamps, and incomplete deliveries must be handled according to documented data-quality rules.

Training and evaluation must use historical delivery outcomes, with validation and testing performed on deliveries not used to train the model. A chronological split should be used to assess performance on future deliveries.

Success will be measured by lower MAE on held-out data and by a reduction in the difference between estimated and actual delivery times in production.

## 2. Feature Mapping: Exactly 10 Features

The following ten features should be considered for predicting delivery duration.

| No. | Feature | Category | Explanation |
|---|---|---|---|
| 1 | Delivery Distance (km) | Numerical | Road-network distance between pickup and delivery locations. |
| 2 | Current Traffic Level | Categorical | Traffic condition classified as Low, Medium, or High. |
| 3 | Weather Condition | Categorical | Conditions such as Clear, Rainy, or Stormy. |
| 4 | Time of Day | Categorical | Time period such as Morning, Afternoon, Evening, or Night. |
| 5 | Day of Week | Categorical | Monday through Sunday, capturing recurring traffic patterns. |
| 6 | Courier Experience (months) | Numerical | Number of months of courier delivery experience. |
| 7 | Package Weight (kg) | Numerical | Measured weight of the package. |
| 8 | Number of Previous Stops | Numerical | Number of delivery stops scheduled before the current delivery on the courier's route. |
| 9 | Restaurant or Merchant Preparation Time (minutes) | Numerical | Elapsed preparation time for the order before it is ready for pickup, if available at prediction time. |
| 10 | Courier-to-Pickup Distance (km) | Numerical | Road-network distance from the courier's current location to the pickup point. |

Categorical features represent discrete labels or groups, while numerical features represent measurable quantities or counts. These categories must be handled consistently during training and inference.

**Feature availability rule:** Only information available when the prediction is generated may be used. For example, actual merchant preparation time measured after the prediction must not be used as an input feature for an earlier prediction. A historical or estimated preparation-time value available at prediction time may be used instead.

All preprocessing rules, category mappings, units, missing-value handling, and feature definitions must be documented and applied consistently in production.

## 3. Real-World Noise and Data Anomalies

### A. GPS Signal Dropouts and Location Errors

Urban areas in Bengaluru may experience GPS inaccuracies because of tall buildings, tunnels, or poor signal reception. A courier's reported location may jump unexpectedly or temporarily disappear.

**Impact:** The system may calculate an incorrect courier-to-pickup distance or estimate an unrealistic route duration. This can cause inaccurate delivery predictions and unstable model behavior.

**Mitigation:** Validate GPS coordinates, detect implausible location jumps, handle missing locations explicitly, and use a documented fallback strategy when reliable location data is unavailable.

### B. Unexpected Traffic Congestion and Road Closures

Accidents, construction, flooding, public events, and temporary road closures can make actual travel times substantially longer than expected.

**Impact:** Historical average travel times may underestimate delivery duration under unusual conditions, increasing prediction errors.

**Mitigation:** Incorporate reliable, time-relevant traffic information where available, monitor prediction errors during unusual events, and evaluate performance across different traffic conditions.

### C. Incorrect or Delayed Delivery Timestamps

Courier applications may record pickup or delivery events late because of network interruptions, offline devices, duplicate requests, or human error.

**Impact:** The recorded delivery duration may not reflect the true elapsed time. Incorrect labels can mislead training and make model evaluation unreliable.

**Mitigation:** Validate timestamp order, identify duplicates, define rules for delayed event synchronization, and exclude or correct records that cannot be verified.

### D. Missing Weather or Traffic Data

External APIs may fail, return stale information, or provide incomplete coverage.

**Impact:** The production system may receive missing or outdated features that were rarely encountered during training, resulting in unreliable predictions or inference failures.

**Mitigation:** Validate feature freshness, monitor missing-value rates, use consistent training and inference preprocessing, and define safe fallback behavior.

## 4. MLflow Experiment Tracking and Engineering Rules

### A. Notebook and Code Management Policy

The team must stop using ambiguous filenames such as `delivery_model_final_v3_really_final.ipynb` as the primary source of production logic.

Jupyter notebooks may be used for exploratory data analysis and prototyping only. Reusable preprocessing, training, evaluation, and inference logic must be extracted into version-controlled Python modules.

The production project must follow a clear structure, such as:

- `notebooks/` — exploratory analysis and temporary experiments.
- `src/data_preprocessing.py` — data validation and preprocessing.
- `src/train.py` — model training.
- `src/evaluate.py` — evaluation and metric calculation.
- `src/predict.py` — production inference.
- `tests/` — automated tests for preprocessing and prediction behavior.
- `requirements.txt` or a locked dependency file — reproducible software dependencies.

Every experiment must identify the source-code version, dataset version, dependency versions, and configuration used. Training must be reproducible from a clean environment without relying on hidden notebook state.

### B. Mandatory MLflow Experiment Tracking

MLflow must be used to track every training experiment. No model may be promoted to production without a corresponding MLflow run containing its required parameters, metrics, artifacts, and version information.

The team must use a consistent experiment name, such as `swiftdrop_delivery_time_prediction`, and assign a unique run to every distinct training configuration.

### C. Required Parameters to Log

Each MLflow run must record:

1. Model algorithm and model configuration.
2. Hyperparameters, including regularization settings, tree depth, number of estimators, or learning rate, as applicable.
3. Training, validation, and test dataset versions or identifiers.
4. Feature list, feature schema, and preprocessing configuration.
5. Train-validation-test split strategy and random seed, where applicable.
6. Source-code Git commit or version identifier.
7. Python version and relevant library versions.
8. Training date, experiment owner, and run description.

### D. Required Metrics to Log

Each run must record:

1. Training MAE in minutes.
2. Validation MAE in minutes.
3. Test MAE in minutes for the final evaluation.
4. Training and validation RMSE.
5. Training and validation loss, where applicable.
6. Prediction-error statistics, including mean error and high-percentile absolute error.
7. Metrics broken down by relevant operating conditions, such as traffic level, weather condition, and time of day.

The test set must not be repeatedly used to tune model hyperparameters. Final test metrics should be reported only after model selection is complete.

### E. Required Artifacts and Promotion Rules

Each run must also store:

- The trained model artifact.
- The fitted preprocessing pipeline or scaler, where applicable.
- Evaluation reports and prediction-error plots.
- Data-quality validation results.
- Feature definitions and the model input schema.
- A record of the acceptance criteria and the final deployment decision.

A model may be promoted only when its evaluation results satisfy predefined business and technical acceptance criteria, its artifacts are reproducible, its input schema is validated, and its inference behavior passes automated tests.

MLflow must be integrated into the team's standard training workflow. Failed experiments must also be recorded rather than silently discarded.

### F. Production Monitoring

After deployment, the team must monitor prediction latency, inference failures, missing or stale features, changes in input-data distributions, and delivery-time prediction errors once actual delivery outcomes become available.

If performance degrades beyond a predefined threshold, the team must investigate the cause, document the incident, and evaluate whether retraining or another corrective action is required. The deployed model version and its corresponding MLflow run must remain traceable.

## Conclusion

SwiftDrop's problem is not simply a poor-performing algorithm. It involves problem definition, data quality, feature availability, reproducibility, and production monitoring.

The recommended strategy is to treat delivery-time estimation as a regression problem, minimize MAE against reliable actual delivery durations, use exactly defined and consistently processed features, address real-world data anomalies, and enforce mandatory MLflow experiment tracking.

Separating exploratory notebooks from production code and establishing reproducible training, testing, deployment, and monitoring practices will help SwiftDrop reduce delivery-time estimation errors and prevent similar engineering failures in the future.
