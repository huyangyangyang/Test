"""
@author:hy
@e-mail:813206372@qq.com
@file: Recommender_System_Project-PyCharm-Baseline_CF_ByALS.py
@time:2022/10/25 18:13
"""
import pandas as pd
import numpy as np


class BaselineCFByALS(object):

    def __init__(self, number_epochs, reg_bu, reg_bi, columns=["uid", "iid", "rating"]):
        # 梯度下降最高迭代次数
        self.number_epochs = number_epochs
        # bu的正则化参数
        self.reg_bu = reg_bu
        # bi的正则化参数
        self.reg_bi = reg_bi
        # 数据集中user-item-rating字段的名称
        self.columns = columns

    def fit(self, dataset):
        '''
        :param dataset:uid,iid,rating
        :return:
        '''
        self.dataset = dataset
        # 用户评分数据
        self.users_ratings = dataset.groupby(self.columns[0]).agg([list])[[self.columns[1], self.columns[2]]]
        # 物品评分数据
        self.items_ratings = dataset.groupby(self.columns[1]).agg([list])[[self.columns[0], self.columns[2]]]
        # 计算全局平均分
        self.global_mean = self.dataset[self.columns[2]].mean()
        # 调用als方法训练模型参数
        self.bu, self.bi = self.als()

    def als(self):
        '''
        利用交替最小二乘法，优化bu,bi的值
        :return:bu,bi
        '''
        # 初始化bu，bi的值，全部设为0
        bu = dict(zip(self.users_ratings.index, np.zeros(len(self.users_ratings))))
        bi = dict(zip(self.items_ratings.index, np.zeros(len(self.items_ratings))))

        for i in range(self.number_epochs):
            print("iter_____%d" % i)
            for iid, uids, ratings in self.items_ratings.itertuples(index=True):
                _sum = 0
                for uid, rating in zip(uids, ratings):
                    _sum += rating - self.global_mean - bu[uid]
                bi[iid] = _sum / (self.reg_bi + len(uids))

            for uid, iids, ratings in self.users_ratings.itertuples(index=True):
                _sum = 0
                for iid, rating in zip(iids, ratings):
                    _sum += rating - self.global_mean - bi[iid]
                bu[uid] = _sum / (self.reg_bu + len(iids))

        return bu, bi

    def predict(self, uid, iid):
        predict_rating = self.global_mean + self.bu[uid] + self.bi[iid]
        return predict_rating


if __name__ == '__main__':
    dtype = [("userId", np.int32), ("movieId", np.int32), ("rating", np.float32)]
    dataset = pd.read_csv('ml-latest-small/ratings.csv', usecols=range(3), dtype=dict(dtype))
    bcf = BaselineCFByALS(20, 25, 15, ["userId", "movieId", "rating"])
    bcf.fit(dataset)

    while True:
        uid = int(input("uid: "))
        iid = int(input("iid: "))
        print(bcf.predict(uid, iid))