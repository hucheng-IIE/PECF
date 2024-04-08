These are some supplementary experiments and charts for the paper "Counterfactual-Augmented Representation Learning for Event Prediction". The paper was submitted to KDD 2024 with the submission number **390**. The paper is currently in the **rebuttal** stage.

## 1. Code and datasets

We have uploaded part of the core code of the model `models.py aggregators.py`, and the remaining data and code will be fully uploaded after the anonymity period.

## 2.Case Study
The figure below shows the visualization of final embeddings learned in each time window using the t-SNE method. The embeddings are obtained from three different models: PECF, PECF without  $`\mathcal{L}_{CF}`$ and $`\mathcal{L}_{dis}`$ (w/o CF) , and tRGCN on the Egypt dataset. The points are colored red and blue based on their labels.

We observe that the representations learned by tRGCN are more scattered and the categories are somewhat mixed. PECF w/o CF performs better than tRGCN, indicating that PECF w/o CF effectively captures graph structure and temporal information. The best representations are learned by PECF, showing that counterfactual results enable the model to deeply understand connection between events and make more accurate event predictions, **demonstrating the validity of identified counterfactual treatments.**
<body>
    <div style="display:flex; justify-content:center; flex-wrap:wrap;">
        <div align=center>
            <img src="https://github.com/hucheng-IIE/PECF/blob/main/case_study/tRGCN.png" alt="Image 1" width="300" height="350"> <img src="https://github.com/hucheng-IIE/PECF/blob/main/case_study/PECF_CF.png" alt="Image 2" width="300" height="350"> <img src="https://github.com/hucheng-IIE/PECF/blob/main/case_study/PECF.png" alt="Image 3" width="350" height="350">
        </div>
    </div>
</body>

## 3.Robustness Tests
We conducted robustness tests on the training and testing sets of the **Egypt and Israel datasets,** using Glean and Cape as baseline methods. Glean performed the best among baseline methods in event prediction tasks on these datasets, while Cape represents the current state-of-the-art method for causal inference-based event prediction.

 - **Robustness to Training Noise**: We consider that noise may exist intraining data due to human or machine errors. To simulate this, we introduce Poisson noise into the training set while keeping the validation and test sets unchanged. We then observe whether the model continues to perform well on the test set despite the added noise in training.
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
<body>
    <div style="display:flex; justify-content:center; flex-wrap:wrap;">
        <div align=center>
            <img src="https://github.com/hucheng-IIE/PECF/blob/main/EG_GT.png" alt="Image 1" width="300" height="350"> 
        </div>
    </div>
</body>

