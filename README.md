<div align="center">

# Collision Guard

**Explore collision-risk classification with synthetic vehicle telemetry and reproducible Jupyter experiments.**

![Python](https://img.shields.io/badge/Python-Jupyter-3776AB?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/ML-scikit--learn-F7931E?logo=scikitlearn&logoColor=white)

[Get started](#quick-start) · [Workflow](#modeling-workflow) · [Results](#experimental-results) · [Limitations](#limitations)

</div>

Collision Guard is an academic machine-learning notebook that compares classifiers for predicting whether simulated driving conditions indicate a collision risk. The experiment uses **500 synthetic telemetry records**, including only **24 positive-risk cases**, to study how severe class imbalance affects model evaluation.

## Features

- Explore vehicle speed, steering, acceleration, braking, proximity, relative speed, weather, and road type.
- Engineer interaction features such as speed-to-distance and braking-to-acceleration ratios.
- Split training and test data before applying **SMOTE** to training samples.
- Compare Random Forest, Gradient Boosting, Logistic Regression, and SVM.
- Evaluate accuracy, precision, recall, F1, and ROC AUC; tune models with `GridSearchCV`.
- Simulate predictions for sampled telemetry records and visualize risk scores.

> [!WARNING]
> This is a **research demonstration**, not a real-time automotive safety system. Its small synthetic dataset and limited collision examples do not establish reliability for autonomous driving or real vehicles.

## Quick start

Requires Python and Jupyter Notebook or JupyterLab.

~~~bash
git clone https://github.com/KOUSHIK1831/Collision-Guard-Autonomous-Vehicle-Risk-Prediction.git
cd Collision-Guard-Autonomous-Vehicle-Risk-Prediction
python -m pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn joblib notebook
jupyter notebook Collision_Risk_Prediction.ipynb
~~~

Run the notebook from top to bottom; keep `collision_risk_dataset.csv` beside it.

## Modeling workflow

```text
Synthetic telemetry CSV
         |
         v
Exploration + feature engineering
         |
         v
Stratified train/test split
         |
         v
Train-only scaling and SMOTE
         |
         v
Four classifier comparison
         |
         v
Tuning + sampled risk simulation
```

**Target:** `risk` (`0` = no reported risk, `1` = collision risk). The dataset has 476 negative and 24 positive records, so accuracy alone can give a misleading impression of quality.

## Experimental results

The project's existing notebook documentation reports:

| Model | Accuracy | Positive-class recall | F1 |
| --- | ---: | ---: | ---: |
| Random Forest | 0.9533 | 0.0000 | 0.0000 |
| Gradient Boosting | 0.9533 | 0.0000 | 0.0000 |
| Logistic Regression | 0.1733 | 0.8571 | 0.0882 |
| SVM | 0.9533 | 0.0000 | 0.0000 |

Logistic Regression detects more positive-risk cases in this split, but at the cost of very low accuracy and precision. The other classifiers miss the positive cases. **No tested model should be considered deployment-ready.** Re-run the notebook for environment-specific results.

## Project structure

| File | Description |
| --- | --- |
| [Collision_Risk_Prediction.ipynb](Collision_Risk_Prediction.ipynb) | Feature engineering, training, comparison, tuning, simulation |
| [collision_risk_dataset.csv](collision_risk_dataset.csv) | Synthetic telemetry used in the experiment |
| `README.md` | Setup and experiment overview |

## Interpreting the metrics

A model that always predicts the majority class would achieve high
accuracy on this dataset while failing to identify collisions.
For a detection task, **positive-class recall** measures how many
risk events are caught, while **precision** measures how often a
risk alert is actually correct.

The reported Logistic Regression experiment has comparatively high
recall (0.8571) but very low precision (approximately 0.0465).
This illustrates a costly false-alert tradeoff; it is not evidence
that the model is suitable for operational collision warnings.
ROC AUC and F1 should be considered alongside the confusion matrix
and the small positive-class sample count.

## Reproducing the experiment

The notebook performs exploration, feature construction, data
splitting, training-sample balancing, model comparison, tuning, and
simulated predictions. To re-run:

1. Keep [`collision_risk_dataset.csv`](collision_risk_dataset.csv)
   in the same directory as the notebook.
2. Install the dependencies shown in [Quick start](#quick-start).
3. Restart the notebook kernel and run all cells in order.
4. Inspect per-class precision/recall, not just overall accuracy.
5. Compare the new results with the reported example metrics.

The code uses `joblib` for a model/scaler serialization workflow.
Treat any generated model artifact as an experimental output, not a
production-grade autonomous-driving control component.

## Next research steps

- Evaluate on larger, independently collected datasets.
- Use repeated stratified evaluation to quantify uncertainty.
- Investigate calibrated probabilities and alert thresholds.
- Separate feature construction, preprocessing, and resampling into
  leak-resistant pipelines for all model comparisons.
- Validate performance on conditions not represented in training.

## Limitations

Evaluation is constrained by the synthetic data, class imbalance, and very small positive test set. A practical safety system would require representative real-world data, stronger validation, calibration, uncertainty handling, and independent safety testing.

> [!TIP]
> Compare **recall and precision together** when reviewing collision-risk predictions. A model that flags every scenario as risky can have high recall without being useful.
