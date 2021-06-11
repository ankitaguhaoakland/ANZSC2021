# Predicting Heart Disease with Machine Learning
The backbone of this study is a dataset from a study of heart disease that has been open to the public for many years. The study collects various measurements on patient health and cardiovascular statistics, and of course makes patient identities anonymous. Developing a Machine Learning Predictive Model that could enhance the predictive power of not only historical patient health data but also with present and future patient health data, with less bias and variance in the model, is the need of the day. For more information on the various Heart Diseases, refer to the [Heart Disease Facts, CDC](https://www.cdc.gov/heartdisease/facts.htm#:~:text=Coronary%20Artery%20Disease,killing%20365%2C914%20people%20in%202017.&text=About%2018.2%20million%20adults%20age,have%20CAD%20(about%206.7%25).&text=About%202%20in%2010%20deaths,less%20than%2065%20years%20old.)

## Data Source
Data is provided courtesy of the Cleveland Heart Disease Database via the [UCI Machine Learning Repository](http://archive.ics.uci.edu/ml/datasets/statlog+(heart)). 
The data that is used for the purpose of this study was hosted at a Competition by [DRIVENDATA](https://www.drivendata.org/competitions/54/machine-learning-with-a-heart/). 

## Data Source Description
There are 14 columns in the dataset, where the patient_id column is a unique and random identifier. The remaining 13 features are described in the section below: <br/>
• slope_of_peak_exercise_st_segment (type: int) : the slope of the peak exercise [ST segment](https://en.wikipedia.org/wiki/ST_segment), an electrocardiography read out indicating quality of blood flow to the heart <br/>
•	thal (type: categorical): results of [thallium stress test](https://www.ucsfbenioffchildrens.org/medical-tests/007201) measuring blood flow to the heart, with possible values normal, fixed defect, reversible defect <br/>
•	resting_blood_pressure (type: int): resting blood pressure <br/>
•	chest_pain_type (type: int): chest pain type (4 values) <br/>
•	num_major_vessels (type: int): number of major vessels (0-3) colored by fluoroscopy <br/>
•	fasting_blood_sugar_gt_120_mg_per_dl (type: binary): fasting blood sugar > 120 mg/dl <br/>
•	resting_ekg_results (type: int): resting electrocardiographic results (values 0,1,2) <br/>
•	serum_cholesterol_mg_per_dl (type: int): serum cholestoral in mg/dl <br/>
•	oldpeak_eq_st_depression (type: float): oldpeak = [ST depression](https://en.wikipedia.org/wiki/ST_depression) induced by exercise relative to rest, a measure of abnormality in electrocardiograms <br/>
•	sex (type: binary): 0: female, 1: male <br/>
•	age (type: int): age in years <br/>
•	max_heart_rate_achieved (type: int): maximum heart rate achieved (beats per minute) <br/>
•	exercise_induced_angina (type: binary): exercise-induced chest pain (0: False, 1: True) <br/>

For further additional information related to the attributes/columns, refer to the [UCI Machine Learning Repository](http://archive.ics.uci.edu/ml/datasets/statlog+(heart)).
