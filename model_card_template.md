# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
Model Name: Census Income Classifier
Algorithm: Logisitc Regression
Frameowkrs/Libraries: scikit-learn, pandas, numpy
Owner: Will Tyndall
Version: v1.0
Training script: train_model.py

## Intended Use
The intended use of this model is educational and demonstrative. It is designed to show an end-to-end machine learning workflow, including data preprocessing, model training, evaluation, and slice-based analysis. It is not intended for high-stakes decision-making such as hiring, lending, housing, or healthcare without additional validation, governance, and human oversight. Users should validate the model on their own data before any deployment.

## Training Data
The training data originates from the UCI Census (Adult) dataset and is loaded from data/census.csv. The target label is salary, which is encoded as a binary outcome (<=50K vs. >50K). The model uses the following categorical features: workclass, education, marital-status, occupation, relationship, race, sex, and native-country. During training, these categorical features are one-hot encoded using a fitted OneHotEncoder, and the label is binarized using a LabelBinarizer.

## Evaluation Data
The evaluation data is a held-out test set that consists of 20% of the original dataset, obtained using a stratified train–test split with a fixed random seed (random_state=42) to preserve class balance. The evaluation set is processed with the same fitted OneHotEncoder and LabelBinarizer used for training to ensure consistent feature mapping.

## Metrics
This model is evaluated using precision, recall, and F1 score on the held-out test set. In the most recent run, the model achieved a precision of 0.7301, a recall of 0.5918, and an F1 score of 0.6538 on the test data. In addition to overall performance, the model’s behavior is examined across subgroups using slice-based metrics. The file slice_output.txt contains precision, recall, and F1 score for each unique value within each categorical feature, along with the count of test examples in that slice. These slice metrics help identify potential disparities in performance across demographic groups.

## Ethical Considerations
This dataset includes sensitive attributes such as race and sex. Even when used for educational purposes, models trained on these features (or their proxies) may reflect historical and societal biases, which can lead to performance disparities across subgroups. Any use of this model in real decision-making contexts should include fairness assessments, human oversight, and careful monitoring. Users should review the slice metrics to identify and address any substantial gaps in precision, recall, or F1 across groups.

## Caveats and Recommendations
This model is trained on the UCI Adult dataset, so I’m treating it as a baseline rather than something I’d deploy without revalidation on our target population. I did hit a convergence warning at the default iteration limit, which is a tuning issue, not a blocker. My immediate next steps are to increase max_iter (for example, to 2000) or switch to the liblinear solver for binary labels, rerun, and log the new metrics. I’ll keep the preprocessing artifacts and feature order locked end-to-end to prevent drift. I will review slice_output.txt and flag any groups with large precision or recall gaps.