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
<body>
    <div style="display:flex; justify-content:center; flex-wrap:wrap;">
        <div align=center>
            <img src="https://github.com/hucheng-IIE/PECF/blob/main/Robustness%20Tests/EG_train.png" alt="Image 1" width="300" height="250"> <img src="https://github.com/hucheng-IIE/PECF/blob/main/Robustness%20Tests/EG_test.png" alt="Image 2" width="300" height="250">
        </div>
    </div>
</body>
## 4.Generalization Tests
