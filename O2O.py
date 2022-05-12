# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import pandas as pd
import xgboost as xgb
data_train = pd.read_csv('ccf_offline_stage1_train.csv')
data_test = pd.read_csv("ccf_offline_stage1_test_revised.csv")
# 测试集数据预处理
def Preprocessing(data):
    data_distance=data['Distance'].fillna(-1)
    data['Distance']=data_distance
    data['date_received'] = pd.to_datetime(data['Date_received'], format='%Y%m%d')
    if 'Date' in [columns for columns in data]:
        data['date'] = pd.to_datetime(data['Date'], format="%Y%m%d")
    return data

# 提取主要的特征
def get_Coupon_Number_Feature(data_set):
    temp_set = data_set.copy()
    temp_set['Coupon_id'] = temp_set['Coupon_id'].map(lambda x: 0 if pd.isnull(x) else int(x))
    temp_set['Date_received'] = temp_set['Date_received'].map(lambda x: 0 if pd.isnull(x) else int(x))
    temp_set['cnt'] = 1
    feature_set = temp_set.copy()
    # 顾客领券数
    keys = ['User_id']
    prefix = 'feature_' + '_'.join(keys) + '_'
    pivot = pd.pivot_table(temp_set, index=keys, values='cnt', aggfunc=len)
    pivot = pd.DataFrame(pivot).rename(columns={'cnt': prefix + 'receive_cnt'}).reset_index()
    feature_set = pd.merge(feature_set, pivot, on=keys, how='left')
    # 顾客领取特定优惠券数量
    keys = ['User_id', 'Coupon_id']
    prefix = 'feature_' + '_'.join(keys) + '_'
    pivot = pd.pivot_table(temp_set, index=keys, values='cnt', aggfunc=len)
    pivot = pd.DataFrame(pivot).rename(columns={'cnt': prefix + 'receive_cnt'}).reset_index()
    feature_set = pd.merge(feature_set, pivot, on=keys, how='left')
    # 顾客当天领取优惠券数量
    keys = ['User_id', 'Date_received']
    prefix = 'feature_' + '_'.join(keys) + '_'
    pivot = pd.pivot_table(temp_set, index=keys, values='cnt', aggfunc=len)
    pivot = pd.DataFrame(pivot).rename(columns={'cnt': prefix + 'receive_cnt'}).reset_index()
    feature_set = pd.merge(feature_set, pivot, on=keys, how='left')
    # 顾客当天领取特定优惠券数量
    keys = ['User_id', 'Date_received', 'Coupon_id']
    prefix = 'feature_' + '_'.join(keys) + '_'
    pivot = pd.pivot_table(temp_set, index=keys, values='cnt', aggfunc=len)
    pivot = pd.DataFrame(pivot).rename(columns={'cnt': prefix + 'receive_cnt'}).reset_index()
    feature_set = pd.merge(feature_set, pivot, on=keys, how='left')
    # 顾客重复领取特定优惠券数量
    keys = ['User_id', 'Date_received', 'Coupon_id']
    prefix = 'feature_' + '_'.join(keys) + '_'
    pivot = pd.pivot_table(temp_set, index=keys, values='cnt', aggfunc=lambda x: 1 if len(x) > 1 else 0)
    pivot = pd.DataFrame(pivot).rename(columns={'cnt': prefix + 'repeat_receive_cnt'}).reset_index()
    feature_set = pd.merge(feature_set, pivot, on=keys, how='left')
    feature_set.drop(['cnt'], axis=1, inplace=True)
    prefix = 'feature_'
    # 商家销售商品数量
    temp_feature = feature_set[['Merchant_id']]
    temp_feature[prefix + 'sales_cnt'] = 1
    temp_feature = temp_feature.groupby('Merchant_id').agg('sum').reset_index()
    feature_set = pd.merge(feature_set, temp_feature, how='left', on='Merchant_id')
    # 商家销售使用了优惠券的商品数量
    temp_feature = feature_set[feature_set.Coupon_id.notnull()][['Merchant_id']]
    temp_feature[prefix + 'sales_coupon_cnt'] = 1
    temp_feature = temp_feature.groupby('Merchant_id').agg('sum').reset_index()
    feature_set = pd.merge(feature_set, temp_feature, how='left', on='Merchant_id')
    # 使用了优惠券进行购买的用户距离商家的最小距离
    temp_feature = feature_set[feature_set.Coupon_id.notnull()][
    ['Merchant_id', 'Distance']].copy()
    temp_feature.rename(columns={'Distance': prefix + 'sales_min_distance'}, inplace=True)
    temp_feature = temp_feature.groupby('Merchant_id').agg('min').reset_index()
    feature_set = pd.merge(feature_set, temp_feature, how='left', on='Merchant_id')
    # 使用了优惠券进行购买的用户距离商家的最大距离
    temp_feature = feature_set[feature_set.Coupon_id.notnull()][
    ['Merchant_id', 'Distance']].copy()
    temp_feature = temp_feature.groupby('Merchant_id').agg('max').reset_index()
    temp_feature.rename(columns={'Distance': prefix + 'sales_max_distance'}, inplace=True)
    feature_set = pd.merge(feature_set, temp_feature, how='left', on='Merchant_id')
    # 使用了优惠券进行购买的用户距离商家的平均距离（领取并消费优惠券的平均距离）
    temp_feature = feature_set[feature_set.Coupon_id.notnull()][
    ['Merchant_id', 'Distance']].copy()
    temp_feature.rename(columns={'Distance': prefix + 'sales_mean_distance'}, inplace=True)
    temp_feature = temp_feature.groupby('Merchant_id').agg('mean').reset_index()
    feature_set = pd.merge(feature_set, temp_feature, how='left', on='Merchant_id')
    # 使用了优惠券进行购买的用户距离商家的中位距离
    temp_feature = feature_set[feature_set.Coupon_id.notnull()][
    ['Merchant_id', 'Distance']].copy()
    temp_feature.rename(columns={'Distance': prefix + 'sales_median_distance'}, inplace=True)
    temp_feature = temp_feature.groupby('Merchant_id').agg('median').reset_index()
    feature_set = pd.merge(feature_set, temp_feature, how='left', on='Merchant_id')
    # 一个顾客在一个商家消费的数量
    temp_feature = feature_set[['User_id', 'Merchant_id']]
    temp_feature[prefix + 'buy_cnt'] = 1
    temp_feature = temp_feature.groupby(['User_id', 'Merchant_id']).agg('sum').reset_index()
    temp_feature.drop_duplicates(inplace=True)
    feature_set = pd.merge(feature_set, temp_feature, how='left', on=['User_id', 'Merchant_id'])
    # 一个顾客在一个商家领取优惠券的数量
    temp_feature = feature_set[feature_set.Coupon_id.notnull()][['User_id', 'Merchant_id']]
    temp_feature[prefix + 'received_cnt'] = 1
    temp_feature = temp_feature.groupby(['User_id', 'Merchant_id']).agg('sum').reset_index()
    temp_feature.drop_duplicates(inplace=True)
    feature_set = pd.merge(feature_set, temp_feature, how='left', on=['User_id', 'Merchant_id'])
    return feature_set

# 提取日期的特征
def get_Date_Feature(data_set):
    temp_set = data_set.copy()
    temp_set['Coupon_id'] = temp_set['Coupon_id'].map(int)
    temp_set['Date_received'] = temp_set['Date_received'].map(int)
    feature_set = temp_set.copy()
    feature_set['weekday'] = feature_set['date_received'].map(lambda x: x.weekday())
    feature_set['isWeekend'] = feature_set['weekday'].map(lambda x: 1 if x == 5 or x == 6 else 0)
    feature_set = pd.concat([feature_set, pd.get_dummies(feature_set['weekday'], prefix='week')], axis=1)
    feature_set.index = range(len(feature_set))
    return feature_set

# 提取优惠券的特征
def get_Coupon_Feature(data_set):
    feature_set = data_set.copy()
    feature_set['isManJian'] = feature_set['Discount_rate'].map(lambda x: 1 if ':' in str(x) else 0)
    feature_set['minPrice'] = feature_set['Discount_rate'].map(lambda x: 0 if ':' not in str(x) else int(str(x).split(':')[0]))
    feature_set['cutPrice'] = feature_set['Discount_rate'].map(lambda x: 0 if ':' not in str(x) else int(str(x).split(':')[1]))
    feature_set['discount_rate'] = feature_set['Discount_rate'].map(lambda x: float(x) 
        if ':' not in str(x)
        else round((float((str(x).split(':')[0])) - float(str(x).split(':')[1]))/float(str(x).split(':')[0]))
    )
    return feature_set

# 提取距离的特征
def get_Distance_Feature(data_set):
    feature_set = data_set.copy()
    feature_set['null_distance'] = feature_set['Distance'].map(lambda x: 1 if x == -1 else 0)
    return feature_set

# 构建数据集
def get_data(feature_field, middle_field, label_field):
    coupon_number_feature = get_Coupon_Number_Feature(label_field)
    date_feature = get_Date_Feature(label_field)
    coupon_feature = get_Coupon_Feature(label_field)
    distance_feature = get_Distance_Feature(label_field)
    common_characters = list(
        set(coupon_number_feature.columns.tolist()) & set(date_feature.columns.tolist()) &
        set(coupon_feature.columns.tolist()) & set(distance_feature.columns.tolist())
    )
    coupon_number_feature.index = range(len(coupon_number_feature))
    coupon_feature.index = range(len(coupon_feature))
    distance_feature.index = range(len(distance_feature))
    data_set = pd.concat([date_feature, coupon_number_feature.drop(common_characters, axis=1),
        coupon_feature.drop(common_characters, axis=1),
        distance_feature.drop(common_characters, axis=1)], axis=1)
    if 'Date' in data_set.columns.tolist():
        data_set.drop(['Merchant_id', 'Discount_rate', 'Date', 'date_received', 'date'], axis=1,
            inplace=True)
        label = data_set['label'].tolist()
        data_set.drop(['label'], axis=1, inplace=True)
        data_set['label'] = label
    else:
        data_set.drop(['Merchant_id', 'Discount_rate', 'date_received'], axis=1, inplace=True)
    data_set['User_id'] = data_set['User_id'].map(int)
    data_set['Coupon_id'] = data_set['Coupon_id'].map(int)
    data_set['Date_received'] = data_set['Date_received'].map(int)
    data_set['Distance'] = data_set['Distance'].map(int)
    if 'label' in data_set.columns.tolist():
        data_set['label'] = data_set['label'].map(int)
    data_set.drop_duplicates(keep='first', inplace=True)
    data_set.index = range(len(data_set))
    return data_set

# 模型训练
def model_xgb(train, test):
    params = {'booster': 'gbtree',
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'silent': 1,
        'eta': 0.01,
        'max_depth': 7,
        'min_child_weight': 5,
        'gamma': 0.2,
        'lambda': 1,
        'colsample_bylevel': 0.7,
        'colsample_bytree': 0.8,
        'subsample': 0.8,
        'scale_pos_weight': 1}
    dtrain = xgb.DMatrix(train.drop(['User_id', 'Coupon_id', 'Date_received', 'label'], axis=1),
                         label=train['label'])
    dtest = xgb.DMatrix(test.drop(['User_id', 'Coupon_id', 'Date_received'], axis=1))
    watchlist = [(dtrain, 'train')]
    model = xgb.train(params, dtrain, num_boost_round=500, evals=watchlist)
    predict = model.predict(dtest)
    predict = pd.DataFrame(predict, columns=['prob'])
    result = pd.concat([test[['User_id', 'Coupon_id', 'Date_received']], predict], axis=1)
    # 特征重要性
    feat_importance = pd.DataFrame(columns=['feature_name', 'importance'])
    feat_importance['feature_name'] = model.get_score().keys()
    feat_importance['importance'] = model.get_score().values()
    feat_importance.sort_values(['importance'], ascending=False, inplace=True)
    return result, feat_importance

# 程序入口
def deal():
    train = Preprocessing(data_train)
    test = Preprocessing(data_test)
    # train集打标
    get_label = train.copy()
    get_label['label'] = list(map(lambda x, y:(y - x).total_seconds() / (3600 * 24),
                             train['date_received'], train['date']))
    train['label'] = list(map(lambda x: 1 if x <= 15 else 0,get_label['label']))
    # 训练集的特征区间
    train_feature_field = train[train['date_received'].isin(pd.date_range('2016/3/16', '2016/5/15'))]
    # 训练集的中间区间
    train_middle_field = train[train['date'].isin(pd.date_range('2016/5/15', '2016/5/30'))]
    # 训练集的标记区间
    train_label_field = train[train['date_received'].isin(pd.date_range('2016/5/31', '2016/7/1'))]
    # 验证集的特征区间
    verify_feature_field = train[train['date_received'].isin(pd.date_range('2016/1/1', '2016/3/1'))]
    # 验证集的中间区间
    verify_middle_field = train[train['date'].isin(pd.date_range('2016/3/1', '2016/3/16'))]
    # 验证集的标记区间
    verify_label_field = train[train['date_received'].isin(pd.date_range('2016/3/16', '2016/4/16'))]
    # 测试集的特征区间
    test_feature_field = train[train['date_received'].isin(pd.date_range('2016/4/17', '2016/6/16'))]
    # 测试集的中间区间
    test_middle_field = train[train['date'].isin(pd.date_range('2016/6/16', '2016/7/1'))]
    # 测试集的标记区间
    test_label_field = test
    train_set = get_data(train_feature_field, train_middle_field, train_label_field)
    verify_set = get_data(verify_feature_field, verify_middle_field, verify_label_field)
    test_set = get_data(test_feature_field, test_middle_field, test_label_field)
    big_train = pd.concat([train_set, verify_set], axis=0)
    result, importance = model_xgb(big_train, test_set)
    result.to_csv('result_test.csv', index=False, header=None)
    print(importance)
    return
# 启动程序
deal()
