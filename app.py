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
        arr_dataset_2 = []
        extension_data = os.path.splitext(file_name)
        if extension_data[1] == ".csv":
            dataset = pd.read_csv(file_name)
        elif extension_data[1] == ".xlsx":
            dataset = pd.read_excel(file_name)
        le = LabelEncoder()
        dataset.Product = le.fit_transform(dataset.Product)
        dataset.Component = le.fit_transform(dataset.Component)
        dataset.Status = le.fit_transform(dataset.Status)
        dataset.Assignee = le.fit_transform(dataset.Assignee)
        dataset.Release = le.fit_transform(dataset.Release)
        dataset.Severity = le.fit_transform(dataset.Severity)
        dataset.isnull().sum()
        columns_of_sheet = dataset[['Product', 'Assignee', 'Status', 'Release', 'Severity']]
        X = np.asarray(columns_of_sheet)
        Y = np.asarray(dataset['Component'])
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
        count = 1
        for line in lines:
            row = {}
            if len(line) > 0:
                row_data = line.split('      ')
                print(row_data)
                print(count, len(row_data))
                print("============================")
                if len(row_data) > 0:
                    if count != 1:
                        if len(row_data) == 6:
                            if len(row_data[0]) > 0:
                                row['Class'] = row_data[0]
                            else:
                                row['Class'] = None
                            num = row_data[1].split()
                            if len(num)>0:
                                Num_data = num[0]
                            else:
                                Num_data = None
                            row['Num_data'] = Num_data

                            Precision = row_data[2].split()
                            if len(Precision) > 0:
                                Precision_data = Precision[0]
                            else:
                                Precision = None
                            row['Precision'] = Precision_data

                            Recall = row_data[3].split()
                            if len(Recall) > 0:
                                Recall_data = Recall[0]
                            else:
                                Recall_data = None
                            row['Recall'] = Recall_data

                            F1_score = row_data[4].split()
                            if len(F1_score) > 0:
                                F1_score_data = F1_score[0]
                            else:
                                F1_score_data = None
                            row['F1_score'] = F1_score_data

                            Support = row_data[5].split()
                            if len(Support) > 0:
                                Support_data = Support[0]
                            else:
                                Support_data = None
                            row['Support'] = Support_data
                        elif len(row_data) == 5:
                            if len(row_data[0]) > 0:
                                row['Class'] = row_data[0]
                            else:
                                row['Class'] = None
                            Num_data = None
                            row['Num_data'] = Num_data

                            Precision = row_data[1].split()
                            if len(Precision) > 0:
                                Precision_data = Precision[0]
                            else:
                                Precision_data = None
                            row['Precision'] = Precision_data

                            Recall = row_data[2].split()
                            if len(Recall) > 0:
                                Recall_data = Recall[0]
                            else:
                                Recall = None
                            row['Recall'] = Recall

                            F1_score = row_data[3].split()
                            if len(F1_score) > 0:
                                F1_score_data = F1_score[0]
                            else:
                                F1_score_data = None
                            row['F1_score'] = F1_score_data

                            Support = row_data[4].split()
                            if len(Support) > 0:
                                Support_data = Support[0]
                            else:
                                Support_data = None
                            row['Support'] = Support_data

                    count = count+1
                    if len(row)>0:
                        report_data.append(row)
        print(len(report_data), report_data)
        dataframe = pd.DataFrame.from_dict(report_data)
        dataframe.to_csv('classification_report.csv', index=False)

        data_csv = pd.read_csv('classification_report.csv')
        headings = data_csv.columns.values
        for index, rows in data_csv.iterrows():
            print(index, rows)
            arr_dataset_2.append(rows.tolist())

        data_set = arr_dataset_2
        #os.remove(file.filename)
        return render_template('predict.html', score=score2*100, headings=headings, data=data_set)


if __name__ == '__main__':
    app.run(debug=True, host='127.0.1.1', port=5000)
