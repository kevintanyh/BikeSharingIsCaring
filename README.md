# Predicting Bike Sharing Capabilities

## About
This is a Mini-Project for SC1015 (Introduction to Data Science and Artificial Intelligence) which focuses on bike sharing from the
[Bike Sharing Dataset Data Set](https://archive.ics.uci.edu/ml/datasets/Bike+Sharing+Dataset).

We apply two models of Ridge Regression and MLP Neural Network onto the bike sharing dataset to accurately predict the total number of bikes rented at any hour, based on the available information and features given in the dataset. 

Please view the following code in this format: 
1. [data_preprocessing.ipynb](https://github.com/kevintanyh/ZXSK/blob/main/data_preprocessing.ipynb)
2. [data_visualisation.ipynb](https://github.com/kevintanyh/ZXSK/blob/main/data_visualisation.ipynb)
3. [data_feature_engineering.ipynb](https://github.com/kevintanyh/ZXSK/blob/main/data_visualisation.ipynb)
4. [model1_Ridge.ipynb](https://github.com/kevintanyh/ZXSK/blob/main/model1_Ridge.ipynb)
5. [model2_MLP.ipynb](https://github.com/kevintanyh/ZXSK/blob/main/model2_MLP.ipynb)
6. [evaluation_conclusion.ipynb](https://github.com/kevintanyh/ZXSK/blob/main/evaluation_conclusion.ipynb)

Additional: [model3_MLP_fulldata.ipynb](https://github.com/kevintanyh/ZXSK/blob/main/model3_MLP_fulldata.ipynb)

## Problem Definition
- Are we able to accurately predict the total number of bikes rented at any hour based on the available information, such as weather and time-related features. To help bike-sharing companies improve bike availability and optimise pricing strategies, which would help them optimise inventory and resources. 
- Which model would be the best to predict it?

## Models Used
- Ridge Regression Model
- MLP Regression Model

## Conclusion
- Total Count (of bike rented at any hours) contained many outliers
- Peak Hour, Temperature and Humidity has the highest correlation to Total Count
- All categorical variables are statistically signficant from one another, hence we isolated them with dummy coding to determine correlation with Total Count
- Hypertuned model improved model performance by a small margin
- Ridge Regression was set to be our Base Model 
- MLP Regression was set to be the second Model to be compared against the base model
- Comparing the best versions for both models, MLP Model 2.2 gave the highest accuracy of 80.6% compared to Ridge Model 2.2 77.8%
- Ultimately, MLP Model 2.2 was concluded to be the most accurate in predicting `total_count` 
- Yes, it is possible to predict the bike rental at any hour with acceptable amount of accuracy to help bike-sharing companies improve bike availability and optimise pricing strategies.

## What did we learn from this project?
1. Feature Engineering 
    - Welch’s ANOVA Test 
    - Dummy Coding
2. Feature Scaling
    - Normalisation 
    - Box-Cox Transformation
3. Model #1: Ridge Regression
4. Model #2 & #3: Multilayer Perceptron (MLP)
5. GitHub Collaboration

## Contributors
- @zxphoon
- @shavonnebay
- @kevintanyh

## References
1. https://towardsdatascience.com/what-is-feature-engineering-importance-tools-and-techniques-for-machine-learning-2080b0269f10
2. https://developers.google.com/machine-learning/data-prep/transform/normalization 
3. https://www.analyticsvidhya.com/blog/2020/04/feature-scaling-machine-learning-normalization-standardization/ 
4. https://leanscape.io/the-box-cox-transformation-what-it-is-and-how-to-use-it/ 
5. https://medium.com/@radoslaw.bialowas/box-cox-transformation-explained-da8450295668 
6. https://statisticsbyjim.com/anova/welchs-anova-compared-to-classic-one-way-anova/ 
7. https://techynotes.medium.com/dummy-variables-in-machine-learning-b3991367bd59
8. https://scholarscompass.vcu.edu/cgi/viewcontent.cgi?article=5026&context=etd 
9. https://www.mygreatlearning.com/blog/what-is-ridge-regression/#:~:text=Ridge%20regression%20is%20a%20model,away%20from%20the%20actual%20values 
10. https://www.analyticsvidhya.com/blog/2017/06/a-comprehensive-guide-for-linear-ridge-and-lasso-regression/ 
11. https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html 
12. https://machinelearningmastery.com/ridge-regression-with-python/ 
13. https://www.analyticsvidhya.com/blog/2016/01/ridge-lasso-regression-python-complete-tutorial/ 
14. https://www.statisticshowto.com/residual-plot/
15. https://otexts.com/fpp2/nnetar.html 
16. https://www.seldon.io/neural-network-models-explained 
17. https://www.mygreatlearning.com/blog/types-of-neural-networks/ 
18. https://towardsdatascience.com/deep-neural-multilayer-perceptron-mlp-with-scikit-learn-2698e77155e 
19. https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPRegressor.html 
20. https://vitalflux.com/sklearn-neural-network-regression-example-mlpregressor/ 
21. https://www.javatpoint.com/multi-layer-perceptron-in-tensorflow
22. https://medium.com/analytics-vidhya/root-mean-square-log-error-rmse-vs-rmlse-935c6cc1802a 
23. https://scholarworks.utep.edu/cs_techrep/1209/#:~:text=Empirical%20studies%20show%20that%20the,of%20the%20data%20for%20training 










