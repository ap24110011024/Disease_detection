# Project Limitations

## Current Limitations

1. The primary experiments were conducted on the Pima Indians Diabetes dataset, with additional validation on the Chronic Kidney Disease (CKD) dataset. Further evaluation on more diverse medical datasets would strengthen the evidence for generalization.

2. The adaptive learning framework was evaluated using sequential batch updates rather than real-world streaming clinical data. Future work can investigate continuous online learning in practical healthcare environments.

3. The proactive detection mechanism relies on rolling-window statistics and threshold-based decision strategies. More advanced temporal models, such as recurrent or transformer-based architectures, could be explored in future work.

4. Hyperparameter optimization was performed using GridSearchCV for selected models. Automated optimization methods such as Bayesian Optimization or Optuna may further improve performance.

5. This work focuses on structured tabular clinical data. Future extensions could integrate multimodal data, including medical imaging, laboratory reports, wearable sensor data, and electronic health records.

## Future Work

- Evaluate the framework on additional disease datasets.
- Investigate continual learning techniques for real-time clinical deployment.
- Compare the proposed adaptive framework with recent deep learning and continual learning approaches.
- Develop a clinician-friendly dashboard for real-time disease risk monitoring.