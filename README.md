# Tabnet benchmark
## Model
Tabnet is proposed as a novel high-performance and interpretable deep tabular data learning architecture. It uses sequential attention to choose which features to reason from at each decision step, enabling local and global interpretability and more efficient learning as the learning capacity is used for the most salient features. Authors state that self-supervised learning for tabular data is significantly improved with unsupervised representation learning when unlabeled data is abundant.
## Task
The task is to predict the probability that an auto insurance policy holder files a claim. The data is anonymized. Evaluation metric is Normalized Gini Coefficient. The dataset consist of 39 columns and about half million rows in training part and about 800 thousands rows in test part. The data is from [Porto Seguro competition](https://www.kaggle.com/c/porto-seguro-safe-driver-prediction/overview). The task is complex and far from being performance-saturated. The best solutions consisted from ensemble of neural networks and gradient boosting trees. It achieve score of 0.2969 (corresponding ROC AUC score 0.6485).
## Results
As baseline model I used random forest. It gets score of 0.269. Tabnet trained on 5 fold cross validation average for folds is 0.248 and tabnet firstly pretrained on unlabeled data and then fine tuned on trained data gets 0.261 Average ensemble of model predictions across folds gets 0.271 and 0.275 respectively. Thus on this task tabnet doesn't outperform baseline model.

As for interpretability, tabnet mostly assign different feature importance across folds. This happen both in supervised and self-supervised settings. It can be explained by problem complexity and inherent randomness of task.
