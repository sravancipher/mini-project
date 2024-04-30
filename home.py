import streamlit as st
import pandas as pd                    # For data manipulation and analysis using DataFrames
import numpy as np                     # For numerical operations
import matplotlib.pyplot as plt        # For creating visualizations
import seaborn as sns                  # For advanced visualizations and statistical graphics
from sklearn.model_selection import train_test_split    # For splitting data into training and testing sets
from sklearn.metrics import confusion_matrix            # Import the confusion_matrix function for evaluating classification results
from sklearn.metrics import classification_report       # Import the classification_report function for detailed classification metrics
from sklearn.linear_model import LogisticRegression    # For Logistic Regression model
from sklearn.ensemble import RandomForestClassifier    # For Random Forest model
from sklearn.tree import DecisionTreeClassifier        # For Decision Tree model
from sklearn.svm import SVC                            # For Support Vector Machine (SVM) model
from sklearn.naive_bayes import GaussianNB            # For Gaussian Naive Bayes model
from sklearn.neighbors import KNeighborsClassifier    # For k-Nearest Neighbors (k-NN) model
# st.sidebar.markdown("""
#    <div>
#             <h1>Health Statistics</h1>
#     </div>
# """,unsafe_allow_html=True)
def app():
    st.markdown("""
            <h1 style='text-align:center;color:brown;'><b>We are here to predict your stress</b></h1>
    """,unsafe_allow_html=True)
    form=st.form("form1")
    sr=form.number_input("Snoring Rate",key="1")
    rr=form.number_input("Respiration Rate",key="2")
    bt=form.number_input("Body Temperature",key="3")
    lm=form.number_input("Limb Movement",key="4")
    bo=form.number_input("Blood Oxygen",key="6")
    em=form.number_input("Eye Movement",key="7")
    sh=form.number_input("Sleeping Hours",key="8")
    hr=form.number_input("Heart Rate",key="9")
    submit=form.form_submit_button("Submit Data")
    st.subheader("Let us predict your stress....")
    
    # col1,col2=st.columns(2)
    # st.sidebar.button("Home")
    # st.sidebar.button("About")
    # st.sidebar.button("Contact Us")
    # Reading the CSV file 'SaYoPillow.csv' and storing the data in a DataFrame called 'data'
    data = pd.read_csv("Stress_Prediction.csv")
    # Displaying the first 5 rows of the dataset
    #st.dataframe(data)
    # Shape of our data
    print("Rows and Columns of the dataset :- ",data.shape)
    # Identifying information about composition and potential data quality
    data.info()
    # Displaying the columns in our dataset
    # data.columns
    # Renaming the columns of the DataFrame for better readability and understanding
    data.columns=['snoring_rate', 'respiration_rate', 'body_temperature', 'limb_movement', 'blood_oxygen','eye_movement', 'sleeping_hours', 'heart_rate', 'stress_level']
    #st.dataframe(data)
    # To show statistical summary of the columns of our data
    data.describe(include="all")
    #checking for null values in the dataframe
    #data.isnull().sum() 
    # To display number of samples on each class
    data['stress_level'].value_counts()
    # Creating a count plot to visualize the distribution of the target variable 'stress_level'
    # using the countplot() function from the seaborn library
    # The 'stress_level' column from the DataFrame 'data' is specified as the x-axis variable
    # sns.countplot(x='stress_level', data=data)

    # Setting the label for the x-axis
    plt.xlabel('Label')

    # Setting the label for the y-axis
    plt.ylabel('Count')

    # Setting the title of the plot
    plt.title('Distribution of the target variable')

    # Displaying the plot
    # plt.show()
    # Histograms for each numerical feature
    data.hist(figsize=(12, 8))
    plt.suptitle("Histograms of Numerical Features", fontsize=16)
    # plt.show()
    # Scatter plots for each numerical feature against 'stress_level'
    for feature in data.columns[:-1]:  # Exclude the target variable 'stress_level'
        plt.figure(figsize=(8, 6))
        sns.scatterplot(data=data, x=feature, y='stress_level', hue='stress_level')
        plt.title(f"{feature} vs. Stress Level")
        plt.xlabel(feature)
        plt.ylabel("Stress Level")
        # plt.show()
    # Violin plots for numerical features based on 'stress_level'
    for feature in data.columns[:-1]:  # Exclude the target variable 'stress_level'
        plt.figure(figsize=(8, 6))
        sns.violinplot(data=data, x='stress_level', y=feature)
        plt.title(f"{feature} Distribution by Stress Level")
        plt.xlabel("Stress Level")
        plt.ylabel(feature)
        # plt.show()
    # Creating a pair plot to visualize pairwise relationships between variables, with 'stress_level' as the hue
    sns.pairplot(data, hue='stress_level')

    # Adding a title to the plot
    plt.title('Pair Plot')

    # Display the pair plot
    # plt.show()
    # Correlation Analysis: Correlation matrix heatmap
    plt.figure(figsize=(10, 8))
    correlation_matrix = data.corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='YlGnBu', fmt='.2f')
    plt.title("Correlation Matrix Heatmap")
    # plt.show()
    # Split the data into features (X) and the target variable (y)
    X = data.drop(['stress_level'], axis=1)
    y = data['stress_level']
    # Split the data into training and testing sets (80% training, 20% testing)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # Display the shapes of the training and testing sets
    print("X_train shape:", X_train.shape)
    print("y_train shape:", y_train.shape)
    print("X_test shape:", X_test.shape)
    print("y_test shape:", y_test.shape)
    # Create an instance of the RandomForestClassifier with hyperparameters
    forest = RandomForestClassifier(n_estimators=500, random_state=1)

    # Train the RandomForestClassifier on the training data
    forest.fit(X_train, y_train.values.ravel())
    # Get the feature importances from the trained RandomForestClassifier
    importances = forest.feature_importances_

    # Loop over each feature and its importance
    for i in range(X_train.shape[1]):
        # Print the feature number, name, and importance score
        print("%2d) %-*s %f" % (i + 1, 30, data.columns[i], importances[i]))
    # Plotting the feature importances as a bar chart
    plt.figure(figsize=(10, 6))
    plt.bar(range(X_train.shape[1]), importances, align='center')
    plt.title('Feature Importance')
    plt.xticks(range(X_train.shape[1]), X_train.columns, rotation=90)
    plt.xlabel('Features')
    plt.ylabel('Importance Score')
    plt.tight_layout()
    plt.show()
    random_forest = RandomForestClassifier(n_estimators=13)
    random_forest.fit(X_train,y_train)
    random_forest.score(X_test,y_test)
    y_predict = random_forest.predict(X_test)

    matrix = confusion_matrix(y_test, y_predict)

    print("Confusion Matrix:")
    print(matrix)
    report = classification_report(y_test, y_predict)

    # Print the classification report
    print("Classification Report:")
    print(report)
    # Creating a logistic regression classifier object with specified parameters
    # using the LogisticRegression class from scikit-learn
    # The max_iter parameter is set to 1000, which determines the maximum number of iterations for convergence
    # The C parameter is set to 0.1, which controls the regularization strength (inverse of the regularization parameter)
    log_reg = LogisticRegression(max_iter=1000, C=0.1)
    # Training the logistic regression classifier using the training data
    # The fit() method is called on the logistic regression object, specifying X_train and y_train as the training data
    log_reg.fit(X_train, y_train)
    # Calculating the accuracy score of the logistic regression model on the test dataset
    # using the score() method of the logistic regression object
    # The X_test and y_test parameters are provided as the test data
    log_reg.score(X_test, y_test)
    # Using the trained logistic regression model to predict the labels for the test dataset
    # using the predict() method of the logistic regression object
    # The X_test parameter is provided as the test data
    # The predicted labels are assigned to the variable y_predict
    y_predict = log_reg.predict(X_test)

    # Calculate the confusion matrix to evaluate the performance of the model
    # using the confusion_matrix() function from scikit-learn
    # The true labels (y_test) and predicted labels (y_predict) are provided as the parameters
    matrix = confusion_matrix(y_test, y_predict)

    print("Confusion Matrix:")
    print(matrix)
    # Predicting Stress Levels
    # To predict stress levels for new data, you can use the 'predict' method of the trained model.
    # For example, let's assume we have new data in a DataFrame called 'new_data':
    if submit:
        new_data = pd.DataFrame([[sr,rr,bt, lm, bo, em, sh,hr]], columns=X.columns)
        # Predict the stress level for the new data
        predicted_stress_level = log_reg.predict(new_data)
        # Dictionary to map integer stress levels to human-readable labels
        stress_level_labels = {
        0: "Low/Normal",
        1: "Medium Low",
        2: "Medium",
        3: "Medium High",
        4: "High"
    }
        # Assuming you already have the 'predicted_stress_level' from the previous code snippet
        predicted_stress_label = stress_level_labels[predicted_stress_level[0]]
        # Display the human-readable label for the predicted stress level
        print("Predicted Stress Label for New Data:",predicted_stress_level[0],"(",predicted_stress_label,")")
        st.markdown("""<h2>Stress Level:</h2>""",unsafe_allow_html=True)
        st.subheader(predicted_stress_label)
    # st.markdown("""<style>
    #             .dvn-scroller.glideDataEditor{
    #             visibility:hidden;
    #             }
    #             </style>""",unsafe_allow_html=True)