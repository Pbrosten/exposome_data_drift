# Exposome Data Drift: Implications for Machine Learning Based Diabetes Prediction


This repository is the result of a Master's Thesis investigation into Traceabiliity. This project was conducted in colaboration with the [BCN-AIM Research Lab](https://www.bcn-aim.org/) and under the supervision of Dr. Karim Lekadir and Marina Camacho. Funding for the work was provided by the European Union’s Horizon 2020 research and innovation program under Grant Agreement Nº 848158 ([EarlyCause](earlycause.europescience.eu)) and Grant Agreement Nº 874739 ([LongITools](www.longitools.org)). The investigation focused on machine learning diabetes risk-predictive models. Specifically, those trained using external exposome data. The repository consists of all Jupyter notebooks used to conduct this exploration. Data was sourced from the [UK Biobank](https://www.ukbiobank.ac.uk/).

## Abstract
Data drift is a problem in machine learning (ML) where characteristics of the input predictors changes over time, leading to model degradation. However, the effects of data drift on ML models built from human exposome data have not been well described yet. This study aimed to investigate data drifts for exposome data in ML models of diabetes risk. 7,521 participants with a diagnosis of diabetes from the UK Biobank, along with a proportional control group from 2006 to 2010 were used to train several baseline ML models for diabetes prediction. A second cohort of 4,007 participants attending the follow-up assessment period from 2012 to 2013 was used to assess potential data drifts over time. When evaluated on the second cohort, significant performance degradation was found in all baseline models (i.e. average precision dropped by 15%, f1-score by 12%, recall by 15%, and precision by 10%). A suite of drift detection tests were run on the best performing baseline models to identify possible signatures of three distinct kinds of data drift: covariate drift, label drift, and concept drift. Utilizing both multivariate and univariate data-distribution based detection methods, covariate drift was identified in features such as Birth Year, BMI, Frequency of Tiredness, and Lack of Education. A comparison of prevalence rates for time-ordered batches of the population found no severe label drift. Nonetheless, gradual label drift could not be ruled out. A model-aware concept drift detection method was employed, monitoring temporal changes in normalized Shapley contributions for the model's input features. This test found drift in abnormal changes in feature contribution when predicting on the second cohort for the Birth Year feature and near alerts in multiple others. This study shows the potential for data drift acting as a driver of model degradation in exposome-based ML models and highlights the need for further research into the traceability of clinical AI/ML solutions.

## Notebooks
### Model Development
[Data Processing and Model Development Notebook](https://github.com/Pbrosten/exposome_data_drift/blob/main/ModelDev_Pipeline.ipynb)

### Drift Analysis
[Covariate Drift Notebook](https://github.com/Pbrosten/exposome_data_drift/blob/main/ModelDev_Pipeline.ipynb)

[Label Drift Notebook](https://github.com/Pbrosten/exposome_data_drift/blob/main/ModelDev_Pipeline.ipynb)

[Concept Drift Notebook](https://github.com/Pbrosten/exposome_data_drift/blob/main/ModelDev_Pipeline.ipynb)
