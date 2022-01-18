import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler


def predict():
    # if request.method == 'POST':
    #     print(file_name)
    #     arr_dataset_2 = []
    #     extension_data = os.path.splitext(file_name)
    #     if extension_data[1] == ".csv":
    #         dataset = pd.read_csv(file_name)
    #     elif extension_data[1] == ".xlsx":
    #         dataset = pd.read_excel(file_name)
    dataset = pd.read_csv("/home/shivani/Downloads/Modified.csv")
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
    print(predict)
    print(np.round(np.clip(g_classifier.predict(X_Test))))
    # predict = g_classifier.fit(predict, X_Test)
    # predict = le.inverse_transform(predict)
    # print(predict)
    report = classification_report(Y_Test, predict)
        # report_data = []
        # lines = report.split('\n')
        # count = 1
        # for line in lines:
        #     row = {}
        #     if len(line) > 0:
        #         row_data = line.split('      ')
        #         # print(row_data)
        #         # print(count, len(row_data))
        #         # print("============================")
        #         if len(row_data) > 0:
        #             if count != 1:
        #                 if len(row_data) == 6:
        #                     if len(row_data[0]) > 0:
        #                         row['Class'] = row_data[0]
        #                     else:
        #                         row['Class'] = '--'
        #                     num = row_data[1].split()
        #                     if len(num)>0:
        #                         Num_data = num[0]
        #                     else:
        #                         Num_data = "--"
        #                     row['Num_data'] = Num_data
        #
        #                     Precision = row_data[2].split()
        #                     if len(Precision) > 0:
        #                         Precision_data = Precision[0]
        #                     else:
        #                         Precision = None
        #                     row['Precision'] = Precision_data
        #
        #                     Recall = row_data[3].split()
        #                     if len(Recall) > 0:
        #                         Recall_data = Recall[0]
        #                     else:
        #                         Recall_data = None
        #                     row['Recall'] = Recall_data
        #
        #                     F1_score = row_data[4].split()
        #                     if len(F1_score) > 0:
        #                         F1_score_data = F1_score[0]
        #                     else:
        #                         F1_score_data = None
        #                     row['F1_score'] = F1_score_data
        #
        #                     Support = row_data[5].split()
        #                     if len(Support) > 0:
        #                         Support_data = Support[0]
        #                     else:
        #                         Support_data = None
        #                     row['Support'] = Support_data
        #                 elif len(row_data) == 5:
        #                     if len(row_data[0]) > 0:
        #                         row['Class'] = row_data[0]
        #                     else:
        #                         row['Class'] = None
        #                     Num_data = None
        #                     row['Num_data'] = Num_data
        #
        #                     Precision = row_data[1].split()
        #                     if len(Precision) > 0:
        #                         Precision_data = Precision[0]
        #                     else:
        #                         Precision_data = None
        #                     row['Precision'] = Precision_data
        #
        #                     Recall = row_data[2].split()
        #                     if len(Recall) > 0:
        #                         Recall_data = Recall[0]
        #                     else:
        #                         Recall = None
        #                     row['Recall'] = Recall
        #
        #                     F1_score = row_data[3].split()
        #                     if len(F1_score) > 0:
        #                         F1_score_data = F1_score[0]
        #                     else:
        #                         F1_score_data = None
        #                     row['F1_score'] = F1_score_data
        #
        #                     Support = row_data[4].split()
        #                     if len(Support) > 0:
        #                         Support_data = Support[0]
        #                     else:
        #                         Support_data = None
        #                     row['Support'] = Support_data
        #
        #             count = count+1
        #             if len(row)>0:
        #                 report_data.append(row)
        # # print(len(report_data), report_data)
        # dataframe = pd.DataFrame.from_dict(report_data)
        # dataframe.to_csv('classification_report.csv', index=False)
        #
        # data_csv = pd.read_csv('classification_report.csv')
        # headings = data_csv.columns.values
        # for index, rows in data_csv.iterrows():
        #     # print(index, rows)
        #     arr_dataset_2.append(rows.tolist())
        #
        # data_set = arr_dataset_2
        # os.remove(file_name)
        # os.remove('classification_report.csv')
        # score_data = round(score2*100, 2)


predict()