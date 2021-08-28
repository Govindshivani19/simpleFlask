import os

import pandas
from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import missingno as msno
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

arr_dataset = []

app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def mypage():
    return render_template('index.html')


@app.route('/read-excel', methods=['GET', 'POST'])
def data():
    if request.method == 'POST':
        file = request.files['upload-file']
        file.save(file.filename)
        extension_data = os.path.splitext(file.filename)
        if extension_data[1] == ".csv":
            data = pd.read_csv(file.filename)
        elif extension_data[1] == ".xlsx":
            data = pd.read_excel(file.filename)
        elif extension_data[1] == ".xls":
            data = pd.read_excel(file.filename)
        headings = data.columns.values
        for index, rows in data.iterrows():
            arr_dataset.append(rows.tolist())

        data_set = arr_dataset
        # os.remove(file.filename)
        return render_template('excel_data.html', headings=headings, data=data_set, file_name= file.filename)


# @app.route('/predict/<file_name>', methods=['GET', 'POST'])
# def predict(file_name):
#     if request.method == 'POST':
#         dataset = pd.read_csv(file_name)
#         print(dataset)
#         le = LabelEncoder()
#         dataset.Product = le.fit_transform(dataset.Product)
#         dataset.Component = le.fit_transform(dataset.Component)
#         dataset.Status = le.fit_transform(dataset.Status)
#         dataset.Assignee = le.fit_transform(dataset.Assignee)
#         dataset.Release = le.fit_transform(dataset.Release)
#         dataset.isnull().sum()
#         columns_of_sheet = dataset[
#             ['Product', 'Component', 'Status', 'Release', 'Severity']]
#         X = np.asarray(columns_of_sheet)
#         Y = np.asarray(dataset['Assignee'])
#         X_Train, X_Test, Y_Train, Y_Test = train_test_split(X, Y,
#                                                             test_size=0.25,
#                                                             random_state=0)
#         sc_X = StandardScaler()
#         X_Train = sc_X.fit_transform(X_Train)
#         X_Test = sc_X.transform(X_Test)
#
#         classifier = SVC(kernel='linear', random_state=5, gamma='auto', C=4)
#         a = classifier.fit(X_Train, Y_Train)
#         print(a)
#
#         Y_Pred = classifier.predict(X_Test)
#         score = classifier.score(X_Train, Y_Train)
#         print(score * 100)
#         print(Y_Pred)
#
#         cm = confusion_matrix(Y_Test, Y_Pred)
#         print(cm)
#
#         return render_template('predict.html')


@app.route('/predict/<file_name>', methods=['GET', 'POST'])
def predict(file_name):
    if request.method == 'POST':
        print(file_name)
        dataset = pd.read_csv(file_name)
        le = LabelEncoder()
        dataset.Product = le.fit_transform(dataset.Product)
        dataset.Component = le.fit_transform(dataset.Component)
        dataset.Status = le.fit_transform(dataset.Status)
        dataset.Assignee = le.fit_transform(dataset.Assignee)
        dataset.Release = le.fit_transform(dataset.Release)
        dataset.isnull().sum()
        columns_of_sheet = dataset[['Product', 'Component', 'Status', 'Release', 'Severity']]
        X = np.asarray(columns_of_sheet)
        Y = np.asarray(dataset['Assignee'])
        X_Train, X_Test, Y_Train, Y_Test = train_test_split(X, Y, test_size=0.25, random_state=0)
        sc_X = StandardScaler()
        X_Train = sc_X.fit_transform(X_Train)
        X_Test = sc_X.transform(X_Test)

        g_classifier = GradientBoostingClassifier(n_estimators=100)
        g_classifier.fit(X_Train, Y_Train)
        score2 = g_classifier.score(X_Test, Y_Test)

        predict = g_classifier.predict(X_Test)
        report = classification_report(Y_Test, predict)
        report_data = []
        lines = report.split('\n')
        count = 0
        for line in lines:
            row = {}
            if len(line) > 0:
                row_data = line.split('      ')
                print(row_data)
                print(count, len(row_data))
                print("============================")
                if len(row_data) > 0:
                    row['class'] = row_data[0]
                    # if count != 1 and count <= len(lines)-3:
                    #     precision = row_data[1].split()
                    #     if len(precision) > 0:
                    #         precision_data = precision[0]
                    #     else:
                    #         precision_data = ""
                    #     row['precision'] = precision_data
                    #
                    #     recall = row_data[3].split()
                    #     if len(recall) > 0:
                    #         recall_data = recall[0]
                    #     else:
                    #         recall_data = ""
                    #     row['recall'] = recall_data
                    #
                    #     print(row_data[4].split())
                    #     f1_score = row_data[3].split()
                    #     if len(f1_score) > 0:
                    #         f1_score_data = f1_score[0]
                    #     else:
                    #         f1_score_data = ""
                    #     row['f1_score'] = f1_score_data
                    #
                    #     support = row_data[5].split()
                    #     if len(support) > 0:
                    #         support_data = support[0]
                    #     else:
                    #         support_data = ""
                    #     row['support'] = support_data
                    # elif count != 1 and count >= len(lines)-3:
                    #     precision = row_data[1].split()
                    #     if len(precision) > 0:
                    #         precision_data = precision[0]
                    #     else:
                    #         precision_data = ""
                    #     row['precision'] = precision_data
                    #
                    #     recall = row_data[2].split()
                    #     if len(recall) > 0:
                    #         recall_data = recall[0]
                    #     else:
                    #         recall_data = ""
                    #     row['recall'] = recall_data
                    #
                    #     print(row_data[3].split())
                    #     f1_score = row_data[3].split()
                    #     if len(f1_score) > 0:
                    #         f1_score_data = f1_score[0]
                    #     else:
                    #         f1_score_data = ""
                    #     row['f1_score'] = f1_score_data
                    #
                    #     support = row_data[4].split()
                    #     if len(support) > 0:
                    #         support_data = support[0]
                    #     else:
                    #         support_data = ""
                    #     row['support'] = support_data
                    count = count+1
                    report_data.append(row)
        print(report_data)
        dataframe = pd.DataFrame.from_dict(report_data)
        dataframe.to_csv('classification_report.csv', index=False)
        return render_template('predict.html', score=score2*100)


if __name__ == '__main__':
    app.run(debug=True, host='127.0.1.1', port=5000)
