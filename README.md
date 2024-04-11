These are some supplementary experiments and charts for the paper "Counterfactual-Augmented Representation Learning for Event Prediction". The paper was submitted to KDD 2024 with the submission number **390**. The paper is currently in the **rebuttal** stage.

## 1. Code and datasets

We have uploaded part of the core code of the model `models.py aggregators.py`, and the remaining data and code will be fully uploaded after the anonymity period.

## 2.Case Study
The figure below shows the visualization of final embeddings learned in each time window using the t-SNE method in the Egypt dataset. The embeddings are obtained from three different models: PECF, PECF without  $`\mathcal{L}_{CF}`$ and $`\mathcal{L}_{dis}`$ (w/o CF), and tRGCN on the Egypt dataset. The points are colored red and blue based on their labels.

We observe that the representations learned by tRGCN are more scattered and the categories are somewhat mixed. PECF w/o CF performs better than tRGCN, indicating that PECF w/o CF effectively captures graph structure and temporal information. The best representations are learned by PECF, showing that counterfactual outcomes enable the model to learn representative embeddings and predict events more accurately, **demonstrating the validity of identified counterfactual treatments**.
<body>
    <div style="display:flex; justify-content:center; flex-wrap:wrap;">
        <div align=center>
            <img src="https://github.com/hucheng-IIE/PECF/blob/main/case_study/tRGCN.png" alt="Image 1" width="300" height="350"> <img src="https://github.com/hucheng-IIE/PECF/blob/main/case_study/PECF_CF.png" alt="Image 2" width="300" height="350"> <img src="https://github.com/hucheng-IIE/PECF/blob/main/case_study/PECF.png" alt="Image 3" width="350" height="350">
            <p style="text-align:center; font-size:20px; font-weight:bold;">Egypt</p>
        </div>
    </div>
</body>

## 3.Robustness Tests
We conducted robustness tests on the training and testing sets of the **Egypt and Israel datasets,** using Glean and Cape as baseline methods. Glean performed the best among baseline methods in event prediction tasks on these datasets, while Cape represents the current state-of-the-art method for causal inference-based event prediction.

 - **Robustness to Training Noise**: We consider that noise may exist in training data due to human or machine errors. To simulate this, we introduce Poisson noise into the training set while keeping the validation and test sets unchanged. We then observe whether the model continues to perform well on the test set despite the added noise in training.
 - **Robustness to Test Noise**: We introduce Poisson noise into the test and validation sets while keeping the training set unchanged. This allows us to observe how the model's performance changes in response to the noise added to the test and validation data.

We set Poisson noise levels to 1, 5, 10, 15, and 20, using the Balanced Accuracy (BACC) metric to assess model performance. The experimental results are shown in the following figure. It can be observed that PECF maintains good performance even in the presence of data noise. **This robustness is attributed to PECF's ability to leverage counterfactual outcomes as supplementary information, enabling a deeper understanding of relationships between events and facilitating robust predictions by the model.**

<body>
    <div style="display:flex; justify-content:center; flex-wrap:wrap;">
        <div align=center>
            <img src="https://github.com/hucheng-IIE/PECF/blob/main/Robustness%20Tests/EG_train.png" alt="Image 1" width="300" height="250"> <img src="https://github.com/hucheng-IIE/PECF/blob/main/Robustness%20Tests/EG_test.png" alt="Image 2" width="300" height="250">
            <p style="text-align:center; font-size:20px; font-weight:bold;">Egypt</p>
        </div>
    </div>
     <div style="display:flex; justify-content:center; flex-wrap:wrap;">
        <div align=center>
            <img src="https://github.com/hucheng-IIE/PECF/blob/main/Robustness%20Tests/IS_train.png" alt="Image 1" width="300" height="250"> <img src="https://github.com/hucheng-IIE/PECF/blob/main/Robustness%20Tests/IS_test.png" alt="Image 2" width="300" height="250">
            <p style="text-align:center; font-size:20px; font-weight:bold;">Israel</p>
        </div>
    </div>
</body>

## 4.Generalization Tests
### 4.1 Different training set sizes
###4.1 Different training set sizes

We reduced the training set size on the Egypt dataset to observe changes in model performance. We set the training set size to 20%, 40%, 60%, and 80% of the dataset, and measured the performance of the PECF model using the evaluation metric BACC. The baselines included Glean, the best baseline method on event prediction in the Egypt dataset, and a state-of-the-art (SOTA) causal inference method for event prediction.

The experimental results are shown in the figure below. We can observe that as the training set size decreases, PECF maintains relatively good performance, **demonstrating that PECF effectively handles data missingness using counterfactual outcomes**.
<body>
    <div style="display:flex; justify-content:center; flex-wrap:wrap;">
        <div align=center>
            <img src="https://github.com/hucheng-IIE/PECF/blob/main/Training_set.png" alt="Image 1" width="300" height="250"> 
            <p style="text-align:center; font-size:20px; font-weight:bold;">Egypt</p>
        </div>
    </div>
</body>

###4.2 Different event types

We conducted a generalization test on the Egypt dataset using three target event types: **protest, appeal, and yield**, and evaluated the performance using the Balanced Accuracy (BACC) metric. The baseline methods included Glean, which performed best on event prediction tasks in the Egypt dataset, and a state-of-the-art (SOTA) method based on causal inference for event prediction.

The experimental results, depicted in the following figure, show that PECF outperformed the other methods, **demonstrating that leveraging counterfactual outcomes can be used to capture the general patterns between data and improve the generalization of the model.**

<body>
    <div style="display:flex; justify-content:center; flex-wrap:wrap;">
        <div align=center>
            <img src="https://github.com/hucheng-IIE/PECF/blob/main/EG_GT.png" alt="Image 1" width="300" height="250"> 
            <p style="text-align:center; font-size:20px; font-weight:bold;">Egypt</p>
        </div>
    </div>
</body>

