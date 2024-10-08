# Machine-Learning-with-Iris-Data-set
## Overview

This project involves the classification of the Iris dataset using various machine learning algorithms. The primary goal is to explore the dataset, preprocess it, and evaluate different classifiers to determine their effectiveness in predicting species of Iris flowers.

## Dataset Description

The Iris dataset consists of 150 samples from three species of Iris flowers: Iris-setosa, Iris-virginica, and Iris-versicolor. Each sample has four features:

- **Sepal Length**: Length of the sepal (cm)
- **Sepal Width**: Width of the sepal (cm)
- **Petal Length**: Length of the petal (cm)
- **Petal Width**: Width of the petal (cm)
- **Species**: Species of the Iris flower (Iris-setosa, Iris-virginica, Iris-versicolor)

The dataset is well-balanced, with 50 samples for each species.

### Species Distribution

```python
datairis['Species'].value_counts()
```

```
Iris-setosa        50
Iris-virginica     50
Iris-versicolor    50
Name: Species, dtype: int64
```

## Data Preparation and Analysis

1. **Load the Dataset**: The Iris dataset is loaded into a DataFrame for analysis.

2. **Data Visualization**:
   - A pair plot is created to visualize the relationships between features, colored by species:
   
   ```python
   import seaborn as sns

   tmp = datairis.drop('Id', axis=1)
   g = sns.pairplot(tmp, hue='Species', markers='*')
   plt.show()
   ```

3. **Normalization**: The features are normalized to ensure that they are on a similar scale, which helps improve the performance of the classifiers.

4. **Splitting the Dataset**: The dataset is split into training and testing sets, initially using a standard split.

5. **Model Training and Evaluation**:
   - **MLP Classifier**: A Multi-layer Perceptron (MLP) classifier is applied to the training data, and accuracy is calculated.
   
6. **Re-splitting the Dataset**: The dataset is split again with a different ratio for training and testing.

7. **Standardization**: StandardScaler is applied to the features to standardize them before training.

8. **Classifier Comparisons**: Several classifiers are trained and evaluated, including:
   - **Support Vector Classifier (SVC)**
   - **Random Forest Classifier**
   - **Decision Tree Classifier**
   - **K-Nearest Neighbors (KNN) Classifier**

   Each classifier's accuracy is calculated and compared.

## Results

The accuracies of the classifiers are summarized as follows:

- **MLP Classifier**: [96.67%]
- **SVC**: [98%]
- **Random Forest Classifier**: [98%]
- **Decision Tree Classifier**: [98%]
- **KNN Classifier**: [98%]

## Conclusion

This project demonstrates the application of various machine learning classifiers on the Iris dataset. By comparing the performance of different models, we gain insights into their effectiveness in classifying Iris species based on feature measurements. Future work could explore additional classifiers or further tuning of hyperparameters to improve accuracy.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.

---

Feel free to fill in the accuracy results and customize any other sections as needed!
