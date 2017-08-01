# -*- coding: utf-8 -*-
"""
Created on Fri Jul 14 14:45:41 2017

@author: Hiram
"""

import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Imputer, OneHotEncoder, RobustScaler, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from xgboost import XGBRegressor
#%%
def split(df, labels):
    return df.loc[:,labels]

def custom_cat(df):
    df = df.replace(['Yes', 'No'], [1,0]) #.fillna(0)
    df = df.replace(['Main campus', 'Not main campus'], [1,0]) #.fillna(0)
    
    df['custom__academics_num_bach'] = df.filter(like='_bach').replace([1,2],[2,1]).sum(axis=1).astype('int')
        
    df['custom__religious_high'] = df.school__religious_affiliation.isin(['myw','cxp','pdf','aiy','thg','zug','sdh','bmv','fxo','qys',
                                             'nnm','onn','ibe','ntl','smi','aai','huu','mix','dpu','fuf','dqz','hmn',
                                             'xds','qzo','mky','hap','fiy','gju','lrj','emi','ddx','jqf']).astype('int')
    df['custom__religious_low'] = df.school__religious_affiliation.isin(['wxa', 'prn', 'qyb', 'nhu','uac', 'rgp', 'iqp',]).astype('int')
    df = df.drop(list(df.filter(like='school__religious_aff')), axis=1)
        
    df['custom__school_state_low'] = df.school__state.isin(['tus', 'nni','noz','ugr','aku','kta','qbv','iju','msx','qid','fen','bbk','sbh','uod',
                                           'gai','idl','gzi','xfa','qua','yyg','xtb','dlg','pgp','krj','bxo','zms','ste',]).astype('int')
    df['custom__school_state_high'] = df.school__state.isin(['prq', 'mig', 'tdb', 'iya', 'wzk', 'afu', 'iyc', 'exw', 'npw', 'rmt','jor','cyf','cmn','ncw','usz',
                                           'tlt','kho','xhl','dhx','nja','ony','rbl','xgy','fyo','das','fjm','hgy']).astype('int')
    df = df.drop(list(df.filter(like='school__state')), axis=1)
    
    df['custom__school_region_low'] = df.school__region_id.isin(['Southwest (AZ, NM, OK, TX)', 
                                 'Southeast (AL, AR, FL, GA, KY, LA, MS, NC, SC, TN, VA, WV)',
                                 'Outlying Areas (AS, FM, GU, MH, MP, PR, PW, VI)',
                                ]).astype('int')
    df['custom__school_region_high'] = df.school__region_id.isin(['New England (CT, ME, MA, NH, RI, VT)',
                                'Mid East (DE, DC, MD, NJ, NY, PA)',
                                'Plains (IA, KS, MN, MO, NE, ND, SD)',
                                'U.S. Service Schools']).astype('int')
    df = df.drop(list(df.filter(like='school__region')), axis=1)
    
    df['custom__carnegie_undergrad_four_year'] = df.school__carnegie_undergrad.str.contains('Four', na=False).astype('int')
    df['custom__carnegie_undergrad_two_year'] = df.school__carnegie_undergrad.str.contains('Two', na=False).astype('int')
    df['custom__carnegie_undergrad_low_transfer'] = df.school__carnegie_undergrad.str.contains('lower', na=False).astype('int')
    df['custom__carnegie_undergrad_high_transfer'] = df.school__carnegie_undergrad.str.contains('higher', na=False).astype('int')
    df['custom__carnegie_undergrad_part_time'] = df.school__carnegie_undergrad.str.contains(' part', na=False).astype('int')
    df['custom__carnegie_undergrad_full_time'] = df.school__carnegie_undergrad.str.contains(' full', na=False).astype('int')   
    df['custom__carnegie_undergrad_inclusive'] = df.school__carnegie_undergrad.str.contains('inclusive', na=False).astype('int')
    df['custom__carnegie_undergrad_selective'] = df.school__carnegie_undergrad.str.contains('selective', na=False).astype('int')

    df['custom__carnegie_undergrad_low'] = df.school__carnegie_undergrad.isin(['Four-year, full-time, inclusive, lower transfer-in', 
                                         'Four-year, higher part-time',
                                         'Four-year, medium full-time, inclusive, lower transfer-in',
                                         'Two-year, medium full-time',
                                         'Two-year, higher full-time',
                                         'Not applicable',
                                         'Two-year, mixed part/full-time',
                                         'Four-year, medium full-time, inclusive, higher transfer-in',
                                         'Two-year, higher part-time',]).astype('int')
    df['custom__carnegie_undergrad_high'] = df.school__carnegie_undergrad.isin(['Four-year, full-time, more selective, lower transfer-in',
                                         'Four-year, full-time, more selective, higher transfer-in',
                                         'Four-year, full-time, selective, higher transfer-in',
                                         'Four-year, full-time, selective, lower transfer-in',
                                         'Four-year, medium full-time, selective, lower transfer-in',
                                         'Four-year, medium full-time, selective, higher transfer-in',
                                         'Not classified (Exclusively Graduate)']).astype('int')
    df = df.drop('school__carnegie_undergrad', axis=1)
    
    df['custom__school_carnegie_size_setting_high'] = df.school__carnegie_size_setting.isin(['Four-year, large, highly residential',
                                             'Four-year, large, primarily residential',
                                             'Four-year, medium, highly residential',
                                             'Four-year, medium, primarily residential',
                                             'Four-year, small, highly residential',
                                             'Exclusively graduate/professional',
                                             'Four-year, small, primarily residential'
                                         ]).astype('int')
    df['custom__school_carnegie_size_setting_low'] = df.school__carnegie_size_setting.isin(['Four-year, very small, primarily nonresidential',
                                             'Not applicable',
                                             'Two-year, large',
                                             'Two-year, medium',
                                             'Two-year, small',
                                             'Two-year, very large',
                                             'Two-year, very small'
                                         ]).astype('int')
    df['custom__carnegie_size_large'] = df.school__carnegie_size_setting.str.contains('large', na=False).astype('int')
    df['custom__carnegie_size_medium'] = df.school__carnegie_size_setting.str.contains('medium', na=False).astype('int')
    df['custom__carnegie_size_small'] = df.school__carnegie_size_setting.str.contains('small', na=False).astype('int')
    df['custom__carnegie_size_highly_residential'] = df.school__carnegie_size_setting.str.contains('highly', na=False).astype('int')
    df['custom__carnegie_size_primarily_residential'] = df.school__carnegie_size_setting.str.contains('primarily', na=False).astype('int')
    df = df.drop('school__carnegie_size_setting', axis=1)
    
    df['custom__carnegie_basic_high'] = df.school__carnegie_basic.isin(['Baccalaureate Colleges: Arts & Sciences Focus',
                                      'Doctoral Universities: Highest Research Activity',
                                      "Master's Colleges & Universities: Larger Programs",
                                      'Doctoral Universities: Higher Research Activity',
                                      "Master's Colleges & Universities: Medium Programs",
                                      'Special Focus Four-Year: Medical Schools & Centers',
                                      "Master's Colleges & Universities: Small Programs",
                                      "Doctoral Universities: Moderate Research Activity",
                                      "Special Focus Four-Year: Law Schools",
                                      "Special Focus Four-Year: Engineering Schools",                                      
                                         ]).astype('int')
    df['custom__carnegie_basic_low'] = df.school__carnegie_basic.isin(['Not applicable',
                                      "Baccalaureate/Associate's Colleges: Mixed Baccalaureate/Associate's",
                                      "Special Focus Two-Year: Health Professions",
                                      "Baccalaureate/Associate's Colleges: Associate's Dominant",
                                      "Associate's Colleges: High Vocational & Technical-High Nontraditional",
                                      "Special Focus Four-Year: Other Technology-Related Schools",
                                      "Associate's Colleges: High Vocational & Technical-Mixed Traditional/Nontraditional",
                                      "Special Focus Two-Year: Other Fields",
                                      "Special Focus Four-Year: Business & Management Schools",
                                      "Associate's Colleges: Mixed Transfer/Vocational & Technical-Mixed Traditional/Nontraditional"
                                     ]).astype('int')
    df['custom__carnegie_basic_bacc'] = df.school__carnegie_basic.str.contains('Baccalaureate', na=False).astype('int')
    df['custom__carnegie_basic_special'] = df.school__carnegie_basic.str.contains('Special', na=False).astype('int')
    df['custom__carnegie_basic_assoc'] = df.school__carnegie_basic.str.contains('Associate', na=False).astype('int')
    df['custom__carnegie_basic_masters'] = df.school__carnegie_basic.str.contains('Master', na=False).astype('int')
    df['custom__carnegie_basic_doc'] = df.school__carnegie_basic.str.contains('Doctoral', na=False).astype('int')
    df = df.drop('school__carnegie_basic', axis=1)
    
    df['custom__school_online'] = df.school__online_only.isin(['Not distance-education only']).astype('int')
    df = df.drop('school__online_only', axis=1)
    
    df['custom__school_locale'] = df.school__locale.isin(['Suburb: Large (outside principal city, in urbanized area with population of 250,000 or more)',
                                            'City: Large (population of 250,000 or more)',
                                            'City: Midsize (population of at least 100,000 but less than 250,000)',
                                            'Rural: Fringe (rural territory up to 5 miles from an urbanized area or up to 2.5 miles from an urban cluster)']).astype('int')
    df['custom__school_locale_city'] = df.school__locale.str.contains('City', na=False).astype('int')
    df['custom__school_locale_suburb'] = df.school__locale.str.contains('Suburb', na=False).astype('int')
    df['custom__school_locale_rural'] = df.school__locale.str.contains('Rural', na=False).astype('int')
    df['custom__school_locale_town'] = df.school__locale.str.contains('Town', na=False).astype('int')
    df = df.drop('school__locale', axis=1)
    
    bachelors = ['biological', 'communication', 'computer', 'education', 'health', 'history', 'mathematics', 'humanities', 'language', 
    'multidiscipline', 'philosophy_religious', 'physical_science', 'psychology', 'social_science', 'visual_performing']
    bachelors = ['academics__program_bachelors_{}'.format(i) for i in bachelors]
    temp = df.loc[:,bachelors] == 1
    df['custom__academics_program_bachelors_good'] = temp.sum(axis=1)
    df['custom__academics_program_bachelors_business'] = df.academics__program_bachelors_business_marketing.isin([1]).astype('int')

    df = df.drop(list(df.filter(like='academics__program')), axis=1)
    #df = df.drop('report_year', axis=1)
    df = df.drop('school__men_only', axis=1)
    
    df = pd.get_dummies(df, dummy_na=True)
    df['custom__sum_missing'] = df.loc[:,list(df.filter(like='nan'))].sum(axis=1)
    df = df.drop(list(df.filter(like='nan')), axis=1)
    #df = df.drop(list(df.filter(like='minority_serving')), axis=1)


    return df

def custom_num(df):
    log_cols = ['cost__title_iv_private_by_income_level_75001_110000', 'cost__title_iv_private_by_income_level_0_30000', 'cost__title_iv_public_by_income_level_110001_plus',
                'completion__completion_cohort_4yr_100nt', 'cost__title_iv_private_by_income_level_75001_110000', 'cost__title_iv_private_by_income_level_110001_plus',
                'aid__median_debt_number_pell_grant', 'aid__loan_principal', 'aid__median_debt_number_pell_grant',
                'student__demographics_age_entry']
    for col in log_cols:
        df['custom__log_{}'.format(col)] = df[col].map(lambda x: np.log(x+1))
    
    log_percentage = ['academics__program_percentage_biological', 'academics__program_percentage_communication', 'academics__program_percentage_english', 'academics__program_percentage_history', 
                      'academics__program_percentage_social_science', 'student__demographics_veteran', 'academics__program_percentage_education',
                      'student__demographics_race_ethnicity_asian', 'student__demographics_avg_family_income',
                      'student__demographics_married', "student__demographics_race_ethnicity_black"]
    for col in log_percentage:
        df['custom__log_{}'.format(col)] = df[col].map(lambda x: np.log(x*100+1))

    df = df.drop(log_cols, axis=1)
    df = df.drop(log_percentage, axis=1)

    df['custom__academics_program_percentage_health'] = df['academics__program_percentage_health'].replace(1, np.nan)
    
    df['custom__sat_0'] = df['admissions__sat_scores_average_by_ope_id']#.fillna(0)
    df['custom__act_0'] = df['admissions__act_scores_midpoint_cumulative']#.fillna(0)

    df = df.drop(list(df.filter(like='scores')), axis=1)
    df.loc[:,list(df.filter(like='scores'))] = df.filter(like='scores').fillna(0)
    df = df.drop(list(df.filter(like='net_price_private')), axis=1)

    return df

def combine(df_num, df_cat):
    return pd.merge(df_num, df_cat, left_index=True, right_index=True)

def impute(train, test):
    avg = train.mean()
    train = train.fillna(avg)
    test = test.fillna(avg)
    return [train, test]

def cluster(train, test):
    aid_clusters = ['aid__cumulative_debt_75th_percentile', 
                'aid__cumulative_debt_90th_percentile',
                'aid__median_debt_completers_monthly_payments', 
                'aid__median_debt_completers_overall']
    cost_clusters = ['cost__attendance_academic_year',
                'cost__tuition_in_state']

    temp_train, temp_test = impute(train, test)
    
    kmeans = KMeans(n_clusters=2, n_jobs=-1)
    
    kmeans.fit(temp_train.loc[:, aid_clusters])
    train['custom__aid_clusters'] = kmeans.labels_ 
    test['custom__aid_clusters'] = kmeans.predict(temp_test.loc[:,aid_clusters])

    kmeans.fit(temp_train.loc[:, cost_clusters])
    train['custom__cost_clusters'] = kmeans.labels_ 
    test['custom__cost_clusters'] = kmeans.predict(temp_test.loc[:,cost_clusters])
    
    return [train, test]

def transformer(train, test):
    def transform(df):
        df_num = split(df, numerical_features)
        df_cat = split(df, categorical_features)
        df_num = custom_num(df_num)
        df_cat = custom_cat(df_cat)
        df = combine(df_num, df_cat)
        return df
    train = transform(train)
    test = transform(test)
    #train, test = impute(train, test)
    train, test = cluster(train, test)
    print('Transform Complete.')
    return [train, test]

#%%

x = pd.read_csv('train_values.csv', index_col='row_id')
test = pd.read_csv('test_values.csv', index_col='row_id')
y = pd.read_csv('train_labels.csv', index_col='row_id')

numerical_features = [i for i in x.columns if len(x[i].unique()) > 4 and x[i].dtype != 'O']
categorical_features = [i for i in x.columns if len(x[i].unique()) < 5 or x[i].dtype == 'O']

#%%
X_train, X_test, y_train, y_test = train_test_split(x, y['repayment_rate'], random_state=80) 
X_train, X_test = transformer(X_train, X_test)

#%% Testing
xgb = XGBRegressor(n_estimators=5000, learning_rate=0.05, max_depth=5, subsample=0.8, colsample_bytree=0.8,
                   min_child_weight=7, reg_alpha=100, nthread=4)
#%%
xgb.fit(X_train, y_train)

print('Training Set Score: {:.3f}'.format(xgb.score(X_train, y_train)))
print('Test Set Score: {:.3f}'.format(xgb.score(X_test, y_test)))
print('RMSE: {}'.format(np.sqrt(mean_squared_error(y_test, xgb.predict(X_test)))))
#%% Output
X_train, X_test = transformer(x, test)
xgb.fit(X_train, y)
test['repayment_rate'] = xgb.predict(X_test)
test[['repayment_rate']].to_csv('results0.csv')

print('Done.')

#%% Grid Search
xgb = XGBRegressor(learning_rate=0.05, n_estimators=4000, max_depth=5, subsample=0.8, colsample_bytree=0.8,
                   min_child_weight=7, reg_alpha=100, nthread=3)
param_grid = { 'reg_alpha':[0, 0.1, 100]}
grid = GridSearchCV(xgb, param_grid=param_grid, cv=5, n_jobs=-1, verbose=1)
grid.fit(X_train, y_train)
print('Best CV Accuracy: {:.3f}'.format(grid.best_score_))
print('Training Set Score: {:.3f}'.format(grid.score(X_train, y_train)))
print('Test Set Score: {:.3f}'.format(grid.score(X_test, y_test)))
print('Best_Parameters: {}'.format(grid.best_params_))
print('RMSE: {}'.format(np.sqrt(mean_squared_error(y_test, grid.predict(X_test)))))

#%% Plot
from matplotlib.pylab import rcParams
from matplotlib import pyplot as plt
rcParams['figure.figsize'] = 12, 50
feat_imp = pd.Series(xgb.booster().get_fscore()).sort_values(ascending=True)
feat_imp.plot(kind='barh', title='Feature Importances')
plt.ylabel('Feature Importance Score')
#%%
import seaborn as sns
rcParams['figure.figsize'] = 10,10
from scipy.stats import pearsonr
temp = pd.merge(X_train[['aid__students_with_any_loan']], #.apply(lambda x: np.log(x*100+1)),
                y[['repayment_rate']], left_index=True, right_index=True).dropna()
sns.regplot(temp.iloc[:,0], temp.iloc[:,1], scatter_kws={'alpha':0.1})
pearsonr(temp.iloc[:,0], temp.iloc[:,1])

#%%
plt.clf()
fig, axes = plt.subplots(1, 2, figsize=(15,6), sharey=True)
ax = axes.ravel()

sns.regplot(X_train['custom__log_academics__program_percentage_english'], y_train, 
  scatter_kws={'alpha':0.1}, ax=ax[0])
ax[0].set_ylim(0,110)
ax[0].set_title('Percentage of Students that are Black', size=16)
ax[0].set_xlabel('Percentage', size=14)
ax[0].set_ylabel('Repayment Rate', size=14)

sns.regplot(x['student__demographics_age_entry'], y['repayment_rate'], 
  scatter_kws={'alpha':0.1}, ax=ax[1])
ax[1].set_ylim(0,110)
ax[1].set_xlim(18,35)
ax[1].set_title('Average Age of Entry', size=16)
ax[1].set_xlabel('Age', size=14)
ax[1].set_ylabel('Repayment Rate', size=14)



#%% XGB CV

import xgboost as xgb
xgb1 = XGBRegressor(
 learning_rate =0.05,
 n_estimators=5000,
 max_depth=5,
 min_child_weight=7,
 reg_alpha=100,
 gamma=0,
 subsample=0.8,
 colsample_bytree=0.8,
 nthread=3,
 scale_pos_weight=1,
 )

xgb_param = xgb1.get_xgb_params()
xgtrain = xgb.DMatrix(X_train.values, label=y_train.values)
xgb.cv(xgb_param, xgtrain, num_boost_round=xgb1.get_params()['n_estimators'], nfold=5,
            metrics='rmse', early_stopping_rounds=50, verbose_eval=100)
#%%
import lightgbm as lgb
#y_train = y_train.values
#y_test = y_test.values
#X_train = X_train.values
#X_test = X_test.values

print('Start training...')
# train
gbm = lgb.LGBMRegressor(objective='regression',
                        num_leaves=127,
                        learning_rate=0.01,
                        n_estimators=2000,
                        num_threads=4,
                        max_bin=500)
gbm.fit(X_train, y_train,
        eval_set=[(X_test, y_test)],
        eval_metric='l1',
        early_stopping_rounds=5)

print('Start predicting...')
# predict
y_pred = gbm.predict(X_test, num_iteration=gbm.best_iteration)
# eval
print('The rmse of prediction is:', mean_squared_error(y_test, y_pred) ** 0.5)