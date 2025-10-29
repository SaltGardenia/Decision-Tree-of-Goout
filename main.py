import pandas as pd
from sklearn import tree
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt


dataset = pd.read_table('tennis.txt', header=None, sep='\t')
print(dataset)

def data2vec(data):
    """
    数据向量化
    :param data:
    :return:
    """
    columns = data.columns[:]
    for i in columns:
        categorical = pd.Categorical(data[i])
        data[i] = categorical.codes
    return data


data = data2vec(dataset)
print(data)
# 独热编码
x = data[[0, 1, 2, 3]]
one_hot = OneHotEncoder(sparse_output=False, categories='auto')
oh_data = one_hot.fit_transform(x)
print(oh_data)


def creat_tree(dataset):
    """
    决策树创建
    :param dataset:
    :return:
    """
    data = dataset.iloc[:, :-1]
    target = dataset.iloc[:, -1]
    tree_model = tree.DecisionTreeClassifier(criterion='gini')
    tree_model.fit(data, target)

    feature_imp = pd.DataFrame({
        '特征': ['天气', '温度', '湿度', '紫外线'],
        '重要性': tree_model.feature_importances_,
    })

    print(feature_imp.sort_values(by='重要性', ascending=False))
    plt.figure(figsize=(16, 10))
    plt.rcParams["font.family"] = ["SimHei"]
    plt.rcParams["axes.unicode_minus"] = False
    plot_tree(tree_model, filled=True, feature_names=['天气', '温度', '湿度', '紫外线'], class_names=['是', '否'], fontsize=12)
    plt.title('天气和出门的关系')
    plt.show()

creat_tree(data)