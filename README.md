# Machine-Learning-for-Survival-Prediction-in-Patients-with-Hepatocellular-Carcinoma-

I. 
Data Set


The dataset used will be the Hepatocellular Carcinoma dataset (HCC dataset) [1], available on Kaggle. This dataset contains real clinical data from 165 patients diagnosed with HCC, collected at the Centro Hospitalar e Universitário de Coimbra (CHUC) by the Internal Medicine A Department. The database contains 49 features—demographic, risk-factor, laboratory, and survival—selected according to the EASL–EORTC Clinical Practice Guidelines (European Association for the Study of the Liver – European Organisation for Research and Treatment of Cancer), which constitute the current state of the art on HCC management [2]. These clinical features—23 quantitative and 26 qualitative—are considered the most significant for clinicians’ decision-making when choosing appropriate therapeutic strategies and predicting outcomes for each patient. Note that not all patients have information for all parameters: missing data account for 10.22% of the entire dataset, and only eight patients have complete information for all fields (4.85%) [3], [4]. In addition, the dataset includes a one-year survival parameter, encoded as a binary variable with values 0 (patient did not survive) and 1 (patient survived).


II. 
Project idea


The main objective of this project is to develop a model that predicts the survival of patients with hepatocellular carcinoma by using data from previously diagnosed patients, through the implementation of Machine Learning techniques. The data provided by the dataset involve several indicators that will be analyzed in order to predict, in a concrete and concise manner, the information to be assessed in the project.
The prior choice of this project was based on the fact that the data provided are relatively recent and refer to Portuguese individuals. Thus, once the model has been developed, it becomes possible to obtain highly useful, relevant, and up-to-date information.


III. 
Software 

This project will be developed mainly in Python, where we will aim to implement the Machine Learning technique that best predicts one-year survival of the HCC patient. The implemented model will create a binary classifier, i.e., with only two possible outcomes: 0 (patient dies) and 1 (patient lives).


IV. 
Relevant Articles

a) “A new cluster-based oversampling method for improving survival prediction of hepatocellular carcinoma patients,” J. Biomed. Inform., vol. 58, pp. 49–59, Dec. 2015, doi: 10.1016/j.jbi.2015.09.012.
b) H.-Y. Shi et al., “Comparison of Artificial Neural Network and Logistic Regression Models for Predicting In-Hospital Mortality after Primary Liver Cancer Surgery,” PLOS ONE, vol. 7, no. 4, p. e35781, Apr. 2012, doi: 10.1371/journal.pone.0035781.
c) A. Forner, J. M. Llovet, and J. Bruix, “Hepatocellular carcinoma,” The Lancet, vol. 379, no. 9822, pp. 1245–1255, Mar. 2012, doi: 10.1016/S0140-6736(11)61347-0.


V. 
References 

[1] “HCC survival data set.” Accessed: November 16, 2023. Available at: https://www.kaggle.com/datasets/mirlei/hcc-survival-data-set
Master’s in Bioengineering – Biomedical Engineering, DACO – Computer-Aided Diagnosis, 2023/2024.
[2] Anon., European Association for the Study of the Liver; European Organisation for Research and Treatment of Cancer, “EASL–EORTC clinical practice guidelines: management of hepatocellular carcinoma,” J. Hepatol. 56 (4) 2012, 908–943.
[3] P. A. Miriam Santos, “HCC Survival.” UCI Machine Learning Repository, 2015. doi: 10.24432/C5TS4S.
[4] “A new cluster-based oversampling method for improving survival prediction of hepatocellular carcinoma patients,” J. Biomed. Inform., vol. 58, pp. 49–59, Dec. 2015, doi: 10.1016/j.jbi.2015.09.012.
