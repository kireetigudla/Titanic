# Implement machine learning algorithms for Titanic Data

## Introduction of Problem

### The Titanic dataset, chronicling the passengers aboard its fateful voyage, presents an opportunity to employ machine learning for survival prediction. This report tackles the binary classification challenge by dissecting and pre-processing the dataset, followed by a critical feature extraction phase to train predictive models.
### Our methodology involves an initial data inspection, cleaning, and normalization to prepare for the application of Naive Bayes and Logistic Regression models. These steps are pivotal in uncovering patterns that reveal the impact of various factors on survival rates.
### Through this analytical endeavor, we aim to not just predict outcomes but to interpret the human narratives behind the data, continuously refining our approach to extract meaningful insights from a historical tragedy.

## Approaches

### Our investigation into the Titanic dataset adopts a structured approach:

### 1. Data Pre-processing: Initial steps involve cleaning to rectify missing values, normalizing data to uniform scales, and pruning irrelevant features. This sets the stage for reliable analysis.

### 2. Exploratory Data Analytics (EDA): We conduct EDA to understand underlying distributions and relationships within the data. Visualization tools like histograms, bar charts, and box plots offer insights into the survival patterns across different passenger classes and demographics.

### 3. Data Transformation: The cleaned dataset undergoes a transformation from a Pandas Data Frame to a Spark Data Frame. We employ `VectorAssembler` to combine features into a single vector, facilitating Spark's machine learning processes.

### 4. Machine Learning Algorithm Implementation: With the data prepped, we apply two predictive models: Naive Bayes, known for its efficiency with categorical data, and Logistic Regression, favoured for binary outcome predictions. These models undergo training, validation, and hyperparameter tuning within a Spark pipeline.

### 5. Performance Evaluation: Using Spark's `MulticlassClassificationEvaluator`, we measure precision, recall, and accuracy, alongside generating a confusion matrix. These metrics help us assess the efficacy of our models and guide further refinement.

### Through these steps, we aim to not only predict survival on the Titanic but also gain a deeper understanding of the factors influencing survival rates.

### Starts with installing spark, importing required libraries, and loading our dataset. 

![image](https://github.com/kireetigudla/Titanic/assets/122108823/7c456f83-4618-414e-8856-5060127a8895)

![image](https://github.com/kireetigudla/Titanic/assets/122108823/41cad597-1f1e-4c37-a363-cb084921224a)

## The bar chart illustrates the distribution of survival outcomes on the Titanic, with a higher number of non-survivors (0) compared to survivors (1).

![image](https://github.com/kireetigudla/Titanic/assets/122108823/ce4d1f36-0cbd-4475-b6bf-6db2d94fa1f4)

## The histogram with a kernel density estimate (KDE) shows the age distribution of Titanic passengers, predominantly skewed towards younger individuals with a peak around the 20-30 age range.

![image](https://github.com/kireetigudla/Titanic/assets/122108823/a88b3887-11e1-48e9-a058-8f26130fb6e2)

## The bar chart represents the survival rate of passengers on the Titanic, categorized by passenger class, showing the highest survival rate for first-class passengers, followed by second and third classes respectively.

![image](https://github.com/kireetigudla/Titanic/assets/122108823/76798697-9bfc-42e3-981a-d9c5c8efdd3a)

### The bar chart illustrates the survival rate on the Titanic by gender, indicating a higher survival rate for females (labelled as '0') compared to males (labelled as '1').
![image](https://github.com/kireetigudla/Titanic/assets/122108823/69b9203d-3f76-459b-9aec-3f24871d749e)

### The bar chart shows the survival rate on the Titanic across different age groups, with the highest rate in the 0-18 range and the lowest in the 60-80 range.
![image](https://github.com/kireetigudla/Titanic/assets/122108823/543a1699-1db1-43cf-8b49-fdc21fb161f0)


### The bar chart shows the survival rates on the Titanic across different passenger classes (Pclass) broken down by age groups, indicating that age and class both had an impact on survival chances.
![image](https://github.com/kireetigudla/Titanic/assets/122108823/df2316e0-ae5e-441f-b1fa-d54155ee9fee)

### The heatmap visualizes the correlation coefficients between different variables in the Titanic dataset, highlighting the strength and direction of the relationships, such as a strong negative correlation between 'Pclass' and 'Fare'.
![image](https://github.com/kireetigudla/Titanic/assets/122108823/a44c51f4-6bc0-4898-b71b-21706218f20b)

### The boxplot visualizes the distribution of ages among the different ticket classes (Pclass) on the Titanic, segmented by the survival outcome.
![image](https://github.com/kireetigudla/Titanic/assets/122108823/ef8b4be1-8ffe-47a4-ad0d-e00c46e8ab85)

## Data Pre-processing Techniques 
![image](https://github.com/kireetigudla/Titanic/assets/122108823/2cb72268-ef3b-4121-ba67-6497bf8cfd40)
### The first screenshot likely shows the `.info()` method output indicating non-null counts and data types for each column in the Titanic Data Frame, while the second probably displays the results of the `. isnull().sum() ` method highlighting the number of missing values per column.
![image](https://github.com/kireetigudla/Titanic/assets/122108823/489868ae-9ce2-4eb3-8738-46a7be0b3637)

### Removing few columns
![image](https://github.com/kireetigudla/Titanic/assets/122108823/c67fa7ec-cb34-4e6b-9a03-f8e822f08f3f)

### Filling missing values
![image](https://github.com/kireetigudla/Titanic/assets/122108823/03284baf-426a-45bd-97f4-e8c8726a35df)

## Data Normalisation
### The steps where the 'Sex' column is encoded into numerical values, and missing values in the 'Embarked' column are imputed with the mean of the non-missing entries.
![image](https://github.com/kireetigudla/Titanic/assets/122108823/02c0204f-e0ef-41cc-a2a3-496c22832b96)

![image](https://github.com/kireetigudla/Titanic/assets/122108823/4b07aee7-4607-446a-ab76-72a86eb04714)

## Finding outliers
### The code calculates the Interquartile Range (IQR) for the 'Age' column of the Titanic dataset, identifies outliers among third-class passengers who did not survive as those outside 1.5 times the IQR beyond the quartiles, and lists their ages.

![image](https://github.com/kireetigudla/Titanic/assets/122108823/34a06337-2e9f-4b11-b787-5153c94b1f41)

### The code snippet filters the Titanic dataset to exclude age outliers among third-class passengers who did not survive, using Boolean indexing to retain only those within the calculated non-outlier age range, and then prints the ages of the remaining passengers.

![image](https://github.com/kireetigudla/Titanic/assets/122108823/97b6b667-b99c-4100-a3fc-733bbe7e18d8)

## Results Aanlysis
### The descriptive statistical analysis on the 'Age' column of the Titanic dataset, providing a count of 891 entries, a mean age of approximately 29.7, a standard deviation of around 13, a minimum age of about 0.42, and a maximum age of 80.
![image](https://github.com/kireetigudla/Titanic/assets/122108823/cdf02c12-3135-47c3-b432-4de868631352)


## Inferential Statistics
### Independent t-test between the ages of survivors and non-survivors of the Titanic, resulting in a t-statistic of -2.0685 and a p-value of 0.0372, suggesting a statistically significant difference in age between the two groups at the 5% significance level.

![image](https://github.com/kireetigudla/Titanic/assets/122108823/01380080-1aed-40df-9c38-4d4a052a1529)

## Correlation Analysis

### The code calculates the Pearson correlation coefficient between the 'Fare' and 'Survived' columns in the Titanic dataset, resulting in a value of approximately 0.257, indicating a weak positive correlation between the fare paid and the chances of survival.

![image](https://github.com/kireetigudla/Titanic/assets/122108823/87387b07-4bee-42db-a033-dfa6643db8f3)


## Predictive Analysis

### The code splits the Titanic dataset into training and test sets, then uses logistic regression to predict survival based on class, sex, age, and fare. The resulting model correctly predicts survival outcomes with about 74.63% accuracy on the test set.

![image](https://github.com/kireetigudla/Titanic/assets/122108823/ecfc1cb8-7f0e-4535-9250-67d014105002)

# Machine Learning

### The code converts a Pandas Data Frame to a Spark Data Frame, applying a predefined schema to ensure proper data types, which is then displayed, confirming the successful transition to a Spark environment for distributed data processing.
![image](https://github.com/kireetigudla/Titanic/assets/122108823/bae60c03-9d02-4ac0-bac2-8b2643ad31f8)

### The code is using Spark's `StringIndexer` to encode the 'Survived' column into numerical labels for machine learning purposes.

![image](https://github.com/kireetigudla/Titanic/assets/122108823/16704e8a-b2a1-4435-a43c-cae3ac1aeb7d)


### This code uses the `VectorAssembler` in Spark to combine multiple columns ('PassengerId', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare') into a single feature vector column named 'features' for machine learning modelling.

![image](https://github.com/kireetigudla/Titanic/assets/122108823/ce0c3763-0128-4c6c-9464-19342a332bec)


### This code snippet constructs a data processing pipeline in PySpark that chains together previously defined pre-processing stages (`clean_up` and `Survived`) to prepare the dataset for machine learning tasks.

![image](https://github.com/kireetigudla/Titanic/assets/122108823/eb95bf55-4f13-4164-ad53-70a0b8c69301)

### This code filters the Spark Data Frame to display only the 'label' and 'features' columns, which likely contain the target variable and assembled feature vector, respectively.

![image](https://github.com/kireetigudla/Titanic/assets/122108823/e46b49b3-5777-43f8-ae1d-634b258c0a0f)


### The code divides the pre-processed dataset into a training set and a test set, with 70% of the data allocated for training and 30% for testing to evaluate the model's performance.

![image](https://github.com/kireetigudla/Titanic/assets/122108823/0f261308-4d22-416e-a6b1-d199efbe4527)

### Applied a trained Naive Bayes classifier to the test data, generating predictions alongside their respective probabilities for each instance.

![image](https://github.com/kireetigudla/Titanic/assets/122108823/cd7a3888-6ed1-4869-bf79-39192acee0ec)

### The code evaluates the accuracy of the Naive Bayes model's predictions using the MulticlassClassificationEvaluator in PySpark, which indicates the model's ability to correctly predict survival with approximately 59.56% accuracy.

![image](https://github.com/kireetigudla/Titanic/assets/122108823/228ec949-3bc4-4d37-a5d9-45355dd56c54)


### Extracts and displays a Data Frame with two columns, 'prediction' and 'label', to compare the model's predictions against the actual labels from the test dataset, providing a direct way to visually assess the prediction accuracy for each instance.

![image](https://github.com/kireetigudla/Titanic/assets/122108823/690dc7c1-0694-4057-9b7e-abf75defafda)

### The code calculates and prints the weighted precision, recall, and accuracy metrics for the predictions, along with the confusion matrix, which provides detailed insight into the performance of the predictive model, indicating that the model has an accuracy of approximately 61.31%.

![image](https://github.com/kireetigudla/Titanic/assets/122108823/0a8d2a49-7fb8-459b-939b-1decdf2cc95d)


### Demonstrates training a logistic regression model on a subset of the Titanic dataset and then testing its performance on a separate test set, with the results showing the predicted versus actual survival outcomes and associated probabilities for each instance.

![image](https://github.com/kireetigudla/Titanic/assets/122108823/506af2fa-d85d-4b22-aa93-9e63715f6ad7)


### Creation of a Data Frame in PySpark by selecting only the 'label' and 'features' columns, which is commonly done as a final step before applying machine learning algorithms; the result shows the first 20 rows of this filtered Data Frame.

![image](https://github.com/kireetigudla/Titanic/assets/122108823/a3f01cce-fc8d-497a-b0cf-743e2295f7d8)

### Using PySpark’s `Multiclass Metrics` library from PySpark to evaluate the performance of a predictive model by calculating precision, recall, accuracy, and generating a confusion matrix based on the predictions compared to the actual labels. The results show a precision of approximately 0.8175, recall of approximately 0.8266, and an overall accuracy of about 0.8175, with a confusion matrix indicating true positive and negative, as well as false positive and negative counts.

![image](https://github.com/kireetigudla/Titanic/assets/122108823/958b407e-c731-406d-9e54-3ecc1fbd72a4)

## Discussion

### 1. Model Performance: Our study highlighted Logistic Regression's superiority over Naive Bayes for the Titanic dataset, achieving around 74.63% accuracy compared to Naive Bayes' 59.56%. This disparity underscores Logistic Regression's adeptness in binary classification and handling mixed data types.

### 2. Model Comparison and Enhancement: The comparison between Naive Bayes and Logistic Regression illuminated the critical role of feature selection and model assumptions. Logistic Regression's superior performance is credited to its better handling of variable interactions and reduced reliance on probabilistic assumptions. Continued improvements through feature engineering and hyperparameter tuning further boosted its accuracy.

### 3. Addressing Challenges: Key challenges like missing data and overfitting were tactically managed. Imputing missing values in 'Age' and 'Embarked', and encoding 'Sex', enhanced data quality. Segmenting data into training and test sets effectively curbed overfitting, ensuring broader applicability of the model.

### 4. Prospects for Future Research: The project opens pathways for advanced modelling techniques. Utilizing ensemble methods, cross-validation, and exploring deep learning in future research could unveil more intricate patterns, potentially elevating prediction precision.



## Conclusions:

### Our research convincingly showcases machine learning's efficacy in dissecting historical data, particularly in revealing the survival dynamics on the Titanic. The standout performance of the Logistic Regression model accentuates the importance of judicious data preparation and model selection tailored to specific analytical requirements.

### While the models yielded commendable results, opportunities for enhancement exist, particularly through more sophisticated machine learning strategies and dataset expansion. This endeavour not only affirms machine learning's value in historical data interpretation but also sets the stage for its application in diverse data landscapes, continually evolving the field of data analytics.

























