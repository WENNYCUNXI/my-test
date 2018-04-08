
# coding: utf-8

import numpy as np
import pandas as pd
from pandas import Series, DataFrame
import matplotlib.pyplot as plt
from pylab import mpl
mpl.rcParams['font.sans-serif'] = ['KaiTi']
mpl.rcParams['font.serif'] = ['KaiTi']

prm_data = pd.read_csv(r'C:\Users\WENNY CUNXI\Desktop\taitannike\a.csv')
prm_data.info()
prm_data.describe()

#总体分析数据中的各个要素
fig = plt.figure(figsize=(15, 15))

plt.subplot2grid((2, 2), (0, 0))
prm_data.Attrition.value_counts().plot(kind='bar')
plt.title(u'公司任职情况图')
plt.ylabel(u'人数')

plt.subplot2grid((2, 2), (0, 1))
prm_data.BusinessTravel.value_counts().plot(kind='bar')
plt.title(u'商务差旅情况图')
plt.ylabel(u'人数')

plt.subplot2grid((2, 2), (1, 0))
prm_data.Department.value_counts().plot(kind='bar')
plt.title(u'部门人数比较')
plt.ylabel(u'人数')

plt.subplot2grid((2, 2), (1, 1))
plt.scatter(prm_data.Attrition, prm_data.DistanceFromHome)
plt.title(u'离家距离与是否离职分布图')
plt.ylabel(u'离家距离')

plt.tight_layout()
plt.savefig(u'1-4图.png', dpi=600)


fig1 = plt.figure(figsize=(10, 10))

plt.subplot2grid((2, 2), (0, 0))
prm_data.Education.value_counts().plot(kind='bar')
plt.title(u'员工教育程度图')
plt.ylabel(u'人数')

plt.subplot2grid((2, 2), (0, 1))
prm_data.EducationField.value_counts().plot(kind='bar')
plt.title(u'员工知识领域分布')
plt.ylabel(u'人数')

plt.subplot2grid((2, 2), (1, 0))
prm_data.EnvironmentSatisfaction.value_counts().plot(kind='bar')
plt.title(u'员工工作满意度分布')
plt.ylabel(u'人数')

plt.subplot2grid((2, 2), (1, 1))
prm_data.Gender.value_counts().plot(kind='bar')
plt.title(u'员工性别')
plt.ylabel(u'人数')

plt.tight_layout()
plt.savefig(u'5-8图.png', dpi=400)

fig2 = plt.figure(figsize=(10, 10))

plt.subplot2grid((2, 2), (0, 0))
prm_data.JobInvolvement.value_counts().plot(kind='bar')
plt.title(u'员工工作投入度')
plt.ylabel(u'人数')

plt.subplot2grid((2, 2), (0, 1))
prm_data.JobLevel.value_counts().plot(kind='bar')
plt.title(u'员工职业级别')
plt.ylabel(u'人数')

plt.subplot2grid((2, 2), (1, 0))
prm_data.JobRole.value_counts().plot(kind='bar')
plt.title(u'员工工作角色')
plt.ylabel(u'人数')

plt.subplot2grid((2, 2), (1, 1))
prm_data.JobSatisfaction.value_counts().plot(kind='bar')
plt.title(u'工作满意度')
plt.ylabel(u'人数')

plt.tight_layout()
plt.savefig(u'9-12图.png', dpi=400)

fig3 = plt.figure(figsize=(10, 10))

plt.subplot2grid((2, 2), (0, 0))
prm_data.MaritalStatus.value_counts().plot(kind='bar')
plt.title(u'员工婚姻状况图')
plt.ylabel(u'人数')

plt.subplot2grid((2, 2), (0, 1))
plt.scatter(prm_data.Attrition, prm_data.MonthlyIncome)
plt.title(u'任职情况与月收入')
plt.ylabel(u'月收入')

plt.subplot2grid((2, 2), (1, 0))
prm_data.NumCompaniesWorked.value_counts().plot(kind='bar')
plt.title(u'员工曾任职公司数')
plt.ylabel(u'人数')

plt.subplot2grid((2, 2), (1, 1))
prm_data.Over18.value_counts().plot(kind='bar')
plt.title(u'员工是否成年')
plt.ylabel(u'人数')

plt.tight_layout()
plt.savefig(u'13-16图.png', dpi=400)

fig4 = plt.figure(figsize=(10, 10))

plt.subplot2grid((2, 2), (0, 0))
prm_data.OverTime.value_counts().plot(kind='bar')
plt.title(u'员工是否加班')
plt.ylabel(u'人数')

plt.subplot2grid((2, 2), (0, 1))
plt.scatter(prm_data.Attrition, prm_data.PercentSalaryHike)
plt.title(u'任职情况与工资提高百分比')
plt.ylabel(u'百分比')

plt.subplot2grid((2, 2), (1, 0))
prm_data.PerformanceRating.value_counts().plot(kind='bar')
plt.title(u'员工绩效')
plt.ylabel(u'人数')

plt.subplot2grid((2, 2), (1, 1))
prm_data.RelationshipSatisfaction.value_counts().plot(kind='bar')
plt.title(u'员工关系满意度')
plt.ylabel(u'人数')

plt.tight_layout()
plt.savefig(u'17-20图.png', dpi=400)

fig5 = plt.figure(figsize=(10, 10))

plt.subplot2grid((2, 2), (0, 0))
prm_data.WorkLifeBalance.value_counts().plot(kind='bar')
plt.title(u'员工工作与生活平衡指数')
plt.ylabel(u'人数')

plt.subplot2grid((2, 2), (0, 1))
prm_data.StockOptionLevel.value_counts().plot(kind='bar')
plt.title(u'员工股票期权')
plt.ylabel(u'人数')

plt.subplot2grid((2, 2), (1, 0))
prm_data.TotalWorkingYears.value_counts().plot(kind='bar')
plt.title(u'员工总工龄')
plt.ylabel(u'人数')

plt.subplot2grid((2, 2), (1, 1))
prm_data.TrainingTimesLastYear.value_counts().plot(kind='bar')
plt.title(u'员工上一年培训时长')
plt.ylabel(u'人数')

plt.tight_layout()
plt.savefig(u'21-24图.png', dpi=400)

fig6 = plt.figure(figsize=(10, 10))

plt.subplot2grid((2, 2), (0, 0))
prm_data.YearsAtCompany.value_counts().plot(kind='bar')
plt.title(u'员工目前在公司工作年数')
plt.ylabel(u'人数')

plt.subplot2grid((2, 2), (0, 1))
prm_data.YearsInCurrentRole.value_counts().plot(kind='bar')
plt.title(u'员工目前职责工作年数')
plt.ylabel(u'人数')

plt.subplot2grid((2, 2), (1, 0))
prm_data.YearsSinceLastPromotion.value_counts().plot(kind='bar')
plt.title(u'员工距离上一次升值时长')
plt.ylabel(u'人数')

plt.subplot2grid((2, 2), (1, 1))
prm_data.YearsWithCurrManager.value_counts().plot(kind='bar')
plt.title(u'员工与管理者共事年数')
plt.ylabel(u'人数')

plt.tight_layout()
plt.savefig(u'25-29图.png', dpi=400)

fig7 = plt.figure()

plt.scatter(prm_data.Attrition, prm_data.Age)
plt.title(u'员工任职情况与年龄图')
plt.ylabel(u'年龄')

plt.tight_layout()
plt.savefig(u'任职情况与年龄.png', dpi=400)

#将要素与任职情况联结分析
fig8 = plt.figure(figsize=(10, 10))

Attrition_0 = prm_data.BusinessTravel[prm_data.Attrition == 0].value_counts()
Attrition_1 = prm_data.BusinessTravel[prm_data.Attrition == 1].value_counts()
df = pd.DataFrame({u'在职':Attrition_0, u'离职':Attrition_1})
df.plot(kind='bar', stacked=True)
plt.title(u'差旅频率与任职情况')
plt.xlabel(u'差旅频率')
plt.ylabel(u'人数')

plt.savefig(u'任职情况与差旅频率.png', dpi=400)

fig9 = plt.figure()

Attrition_0 = prm_data.Department[prm_data.Attrition == 0].value_counts()
Attrition_1 = prm_data.Department[prm_data.Attrition == 1].value_counts()
df = pd.DataFrame({u'在职':Attrition_0, u'离职':Attrition_1})
df.plot(kind='bar', stacked=True)
plt.title(u'任职部门与任职情况')
plt.xlabel(u'任职部门')
plt.ylabel(u'人数')


plt.savefig(u'任职情况与任职部门.png', dpi=400)

fig10 = plt.figure()

Attrition_0 = prm_data.Education[prm_data.Attrition == 0].value_counts()
Attrition_1 = prm_data.Education[prm_data.Attrition == 1].value_counts()
df = pd.DataFrame({u'在职':Attrition_0, u'离职':Attrition_1})
df.plot(kind='bar', stacked=True)
plt.title(u'受教育情况与任职情况')
plt.xlabel(u'受教育情况')
plt.ylabel(u'人数')

plt.savefig(u'受教育情况与任职情况.png', dpi=400)

fig11 = plt.figure()

Attrition_0 = prm_data.EducationField[prm_data.Attrition == 0].value_counts()
Attrition_1 = prm_data.EducationField[prm_data.Attrition == 1].value_counts()
df = pd.DataFrame({u'在职':Attrition_0, u'离职':Attrition_1})
df.plot(kind='bar', stacked=True)
plt.title(u'专业领域与任职情况')
plt.xlabel(u'专业领域')
plt.ylabel(u'人数')

plt.savefig(u'专业领域与任职情况.png', dpi=400)

fig11 = plt.figure()

Attrition_0 = prm_data.EnvironmentSatisfaction[prm_data.Attrition == 0].value_counts()
Attrition_1 = prm_data.EnvironmentSatisfaction[prm_data.Attrition == 1].value_counts()
df = pd.DataFrame({u'在职':Attrition_0, u'离职':Attrition_1})
df.plot(kind='bar', stacked=True)
plt.title(u'工作满意度与任职情况')
plt.xlabel(u'工作满意度')
plt.ylabel(u'人数')

plt.savefig(u'工作满意度与任职情况.png', dpi=400)

fig = plt.figure()

Attrition_0 = prm_data.Gender[prm_data.Attrition == 0].value_counts()
Attrition_1 = prm_data.Gender[prm_data.Attrition == 1].value_counts()
df = pd.DataFrame({u'在职':Attrition_0, u'离职':Attrition_1})
df.plot(kind='bar', stacked=True)
plt.title(u'性别与任职情况')
plt.xlabel(u'性别')
plt.ylabel(u'人数')

plt.savefig(u'性别与任职情况.png', dpi=400)

fig = plt.figure()

Attrition_0 = prm_data.JobInvolvement[prm_data.Attrition == 0].value_counts()
Attrition_1 = prm_data.JobInvolvement[prm_data.Attrition == 1].value_counts()
df = pd.DataFrame({u'在职':Attrition_0, u'离职':Attrition_1})
df.plot(kind='bar', stacked=True)
plt.title(u'工作投入与任职情况')
plt.xlabel(u'工作投入')
plt.ylabel(u'人数')

plt.savefig(u'工作投入与任职情况.png', dpi=400)

fig = plt.figure()

Attrition_0 = prm_data.JobLevel[prm_data.Attrition == 0].value_counts()
Attrition_1 = prm_data.JobLevel[prm_data.Attrition == 1].value_counts()
df = pd.DataFrame({u'在职':Attrition_0, u'离职':Attrition_1})
df.plot(kind='bar', stacked=True)
plt.title(u'职业级别与任职情况')
plt.xlabel(u'职业级别')
plt.ylabel(u'人数')

plt.savefig(u'职业级别与任职情况.png', dpi=400)


fig = plt.figure()

Attrition_0 = prm_data.JobRole[prm_data.Attrition == 0].value_counts()
Attrition_1 = prm_data.JobRole[prm_data.Attrition == 1].value_counts()
df = pd.DataFrame({u'在职':Attrition_0, u'离职':Attrition_1})
df.plot(kind='bar', stacked=True)
plt.title(u'工作角色与任职情况')
plt.xlabel(u'工作角色')
plt.ylabel(u'人数')

plt.savefig(u'工作角色与任职情况.png', dpi=400)

fig = plt.figure()

Attrition_0 = prm_data.JobSatisfaction[prm_data.Attrition == 0].value_counts()
Attrition_1 = prm_data.JobSatisfaction[prm_data.Attrition == 1].value_counts()
df = pd.DataFrame({u'在职':Attrition_0, u'离职':Attrition_1})
df.plot(kind='bar', stacked=True)
plt.title(u'工作满意度与任职情况')
plt.xlabel(u'工作满意度')
plt.ylabel(u'人数')

plt.savefig(u'工作满意度与任职情况.png', dpi=400)

fig = plt.figure()

Attrition_0 = prm_data.MaritalStatus[prm_data.Attrition == 0].value_counts()
Attrition_1 = prm_data.MaritalStatus[prm_data.Attrition == 1].value_counts()
df = pd.DataFrame({u'在职':Attrition_0, u'离职':Attrition_1})
df.plot(kind='bar', stacked=True)
plt.title(u'婚姻状况与任职情况')
plt.xlabel(u'婚姻状况')
plt.ylabel(u'人数')

plt.savefig(u'婚姻状况与任职情况.png', dpi=400)

fig = plt.figure()

Attrition_0 = prm_data.NumCompaniesWorked[prm_data.Attrition == 0].value_counts()
Attrition_1 = prm_data.NumCompaniesWorked[prm_data.Attrition == 1].value_counts()
df = pd.DataFrame({u'在职':Attrition_0, u'离职':Attrition_1})
df.plot(kind='bar', stacked=True)
plt.title(u'工作公司数与任职情况')
plt.xlabel(u'工作公司数')
plt.ylabel(u'人数')

plt.savefig(u'工作公司数与任职情况.png', dpi=400)

fig = plt.figure()

Attrition_0 = prm_data.OverTime[prm_data.Attrition == 0].value_counts()
Attrition_1 = prm_data.OverTime[prm_data.Attrition == 1].value_counts()
df = pd.DataFrame({u'在职':Attrition_0, u'离职':Attrition_1})
df.plot(kind='bar', stacked=True)
plt.title(u'是否加班与任职情况')
plt.xlabel(u'是否加班')
plt.ylabel(u'人数')

plt.savefig(u'是否加班与任职情况.png', dpi=400)

fig = plt.figure()

Attrition_0 = prm_data.PercentSalaryHike[prm_data.Attrition == 0].value_counts()
Attrition_1 = prm_data.PercentSalaryHike[prm_data.Attrition == 1].value_counts()
df = pd.DataFrame({u'在职':Attrition_0, u'离职':Attrition_1})
df.plot(kind='bar', stacked=True)
plt.title(u'工资提高百分比与任职情况')
plt.xlabel(u'工资提高百分比')
plt.ylabel(u'人数')

plt.savefig(u'工资提高百分比与任职情况.png', dpi=400)

fig = plt.figure()

Attrition_0 = prm_data.PerformanceRating[prm_data.Attrition == 0].value_counts()
Attrition_1 = prm_data.PerformanceRating[prm_data.Attrition == 1].value_counts()
df = pd.DataFrame({u'在职':Attrition_0, u'离职':Attrition_1})
df.plot(kind='bar', stacked=True)
plt.title(u'绩效与任职情况')
plt.xlabel(u'绩效投入')
plt.ylabel(u'人数')

plt.savefig(u'绩效与任职情况.png', dpi=400)

fig = plt.figure()

Attrition_0 = prm_data.RelationshipSatisfaction[prm_data.Attrition == 0].value_counts()
Attrition_1 = prm_data.RelationshipSatisfaction[prm_data.Attrition == 1].value_counts()
df = pd.DataFrame({u'在职':Attrition_0, u'离职':Attrition_1})
df.plot(kind='bar', stacked=True)
plt.title(u'关系满意度与任职情况')
plt.xlabel(u'关系满意度')
plt.ylabel(u'人数')

plt.savefig(u'关系满意度与任职情况.png', dpi=400)

fig = plt.figure()

Attrition_0 = prm_data.StockOptionLevel[prm_data.Attrition == 0].value_counts()
Attrition_1 = prm_data.StockOptionLevel[prm_data.Attrition == 1].value_counts()
df = pd.DataFrame({u'在职':Attrition_0, u'离职':Attrition_1})
df.plot(kind='bar', stacked=True)
plt.title(u'股票期权与任职情况')
plt.xlabel(u'股票期权')
plt.ylabel(u'人数')

plt.savefig(u'股票期权与任职情况.png', dpi=400)

fig = plt.figure(figsize=(10, 10))

Attrition_0 = prm_data.TotalWorkingYears[prm_data.Attrition == 0].value_counts()
Attrition_1 = prm_data.TotalWorkingYears[prm_data.Attrition == 1].value_counts()
df = pd.DataFrame({u'在职':Attrition_0, u'离职':Attrition_1})
df.plot(kind='bar', stacked=True)
plt.title(u'总工时与任职情况')
plt.xlabel(u'总工时')
plt.ylabel(u'人数')

plt.savefig(u'总工时与任职情况.png', dpi=400)

fig = plt.figure()

Attrition_0 = prm_data.TrainingTimesLastYear[prm_data.Attrition == 0].value_counts()
Attrition_1 = prm_data.TrainingTimesLastYear[prm_data.Attrition == 1].value_counts()
df = pd.DataFrame({u'在职':Attrition_0, u'离职':Attrition_1})
df.plot(kind='bar', stacked=True)
plt.title(u'上一年培训时长与任职情况')
plt.xlabel(u'上一年培训时长')
plt.ylabel(u'人数')

plt.savefig(u'上一年培训时长与任职情况.png', dpi=400)

fig = plt.figure()

Attrition_0 = prm_data.WorkLifeBalance[prm_data.Attrition == 0].value_counts()
Attrition_1 = prm_data.WorkLifeBalance[prm_data.Attrition == 1].value_counts()
df = pd.DataFrame({u'在职':Attrition_0, u'离职':Attrition_1})
df.plot(kind='bar', stacked=True)
plt.title(u'生活工作平衡与任职情况')
plt.xlabel(u'生活工作平衡')
plt.ylabel(u'人数')

plt.savefig(u'生活工作平衡与任职情况.png', dpi=400)

fig = plt.figure()

Attrition_0 = prm_data.YearsAtCompany[prm_data.Attrition == 0].value_counts()
Attrition_1 = prm_data.YearsAtCompany[prm_data.Attrition == 1].value_counts()
df = pd.DataFrame({u'在职':Attrition_0, u'离职':Attrition_1})
df.plot(kind='bar', stacked=True)
plt.title(u'公司工作年数与任职情况')
plt.xlabel(u'公司工作年数')
plt.ylabel(u'人数')

plt.savefig(u'公司工作年数与任职情况.png', dpi=400)

fig = plt.figure()

Attrition_0 = prm_data.YearsInCurrentRole[prm_data.Attrition == 0].value_counts()
Attrition_1 = prm_data.YearsInCurrentRole[prm_data.Attrition == 1].value_counts()
df = pd.DataFrame({u'在职':Attrition_0, u'离职':Attrition_1})
df.plot(kind='bar', stacked=True)
plt.title(u'工作职责年数与任职情况')
plt.xlabel(u'工作职责年数')
plt.ylabel(u'人数')

plt.savefig(u'工作职责年数与任职情况.png', dpi=400)

fig = plt.figure()

Attrition_0 = prm_data.YearsSinceLastPromotion[prm_data.Attrition == 0].value_counts()
Attrition_1 = prm_data.YearsSinceLastPromotion[prm_data.Attrition == 1].value_counts()
df = pd.DataFrame({u'在职':Attrition_0, u'离职':Attrition_1})
df.plot(kind='bar', stacked=True)
plt.title(u'距离上一次升职时长与任职情况')
plt.xlabel(u'距离上一次升职时长')
plt.ylabel(u'人数')

plt.savefig(u'距离上一次升值时长与任职情况.png', dpi=400)

fig = plt.figure()

Attrition_0 = prm_data.YearsWithCurrManager[prm_data.Attrition == 0].value_counts()
Attrition_1 = prm_data.YearsWithCurrManager[prm_data.Attrition == 1].value_counts()
df = pd.DataFrame({u'在职':Attrition_0, u'离职':Attrition_1})
df.plot(kind='bar', stacked=True)
plt.title(u'跟目前的管理者共事年数与任职情况')
plt.xlabel(u'跟目前的管理者共事年数')
plt.ylabel(u'人数')

plt.savefig(u'跟目前的管理者共事年数与任职情况.png', dpi=400)

#对数值型特征进行特征因子化
dummies_BusinessTravel = pd.get_dummies(prm_data['BusinessTravel'], prefix='BusinessTravel')
dummies_Department = pd.get_dummies(prm_data['Department'], prefix='Department')
dummies_Education = pd.get_dummies(prm_data['Education'], prefix='Education')
dummies_EducationField = pd.get_dummies(prm_data['EducationField'], prefix='EducationField')
dummies_EnvironmentSatisfaction = pd.get_dummies(prm_data['EnvironmentSatisfaction'], prefix='EnvironmentSatisfaction')
dummies_Gender = pd.get_dummies(prm_data['Gender'], prefix='Gender')
dummies_JobInvolvement = pd.get_dummies(prm_data['JobInvolvement'], prefix='JobInvolvement')
dummies_JobLevel = pd.get_dummies(prm_data['JobLevel'], prefix='JobLevel')
dummies_JobRole = pd.get_dummies(prm_data['JobRole'], prefix='JobRole')
dummies_JobSatisfaction = pd.get_dummies(prm_data['JobSatisfaction'], prefix='JobSatisfaction')
dummies_MaritalStatus = pd.get_dummies(prm_data['MaritalStatus'], prefix='MaritalStatus')
dummies_NumCompaniesWorked = pd.get_dummies(prm_data['NumCompaniesWorked'], prefix='NumCompaniesWorked')
dummies_OverTime = pd.get_dummies(prm_data['OverTime'], prefix='OverTime')
dummies_PerformanceRating = pd.get_dummies(prm_data['PerformanceRating'], prefix='PerformanceRating')
dummies_RelationshipSatisfaction = pd.get_dummies(prm_data['RelationshipSatisfaction'], prefix='RelationshipSatisfaction')
dummies_StockOptionLevel = pd.get_dummies(prm_data['StockOptionLevel'], prefix='StockOptionLevel')
dummies_TrainingTimesLastYear = pd.get_dummies(prm_data['TrainingTimesLastYear'], prefix='TrainingTimesLastYear')
dummies_WorkLifeBalance = pd.get_dummies(prm_data['WorkLifeBalance'], prefix='WorkLifeBalance')

#将因子化后的数据与原数据合并
df = pd.concat([prm_data, dummies_BusinessTravel, dummies_Department, dummies_Education, dummies_EducationField, dummies_EnvironmentSatisfaction,
               dummies_Gender, dummies_JobInvolvement, dummies_JobLevel, dummies_JobRole, dummies_JobSatisfaction, dummies_MaritalStatus, dummies_NumCompaniesWorked,
               dummies_OverTime, dummies_PerformanceRating, dummies_RelationshipSatisfaction, dummies_StockOptionLevel, dummies_TrainingTimesLastYear, dummies_WorkLifeBalance],
               axis=1)

#删掉特征因子化前的数据
df.drop(['BusinessTravel', 'Department', 'Education', 'EducationField', 'EnvironmentSatisfaction', 'Gender', 'JobInvolvement', 'JobLevel',
        'JobRole', 'JobSatisfaction', 'MaritalStatus', 'NumCompaniesWorked', 'OverTime', 'PerformanceRating', 'RelationshipSatisfaction',
        'StockOptionLevel', 'TrainingTimesLastYear', 'WorkLifeBalance'], axis=1, inplace=True)
df

#将数值变化较大的数据特征化到[-1, 1]之间
import sklearn.preprocessing as preprocessing
scaler = preprocessing.StandardScaler()
scaler_param = scaler.fit(df[['Age', 'MonthlyIncome', 'PercentSalaryHike', 'TotalWorkingYears', 'YearsAtCompany', 'YearsInCurrentRole', 
                                  'YearsSinceLastPromotion', 'YearsWithCurrManager', 'DistanceFromHome']])
df['Age_scaler'] = scaler.fit_transform(df[['Age', 'MonthlyIncome', 'PercentSalaryHike', 'TotalWorkingYears', 'YearsAtCompany', 'YearsInCurrentRole', 
                                            'YearsSinceLastPromotion', 'YearsWithCurrManager', 'DistanceFromHome']], scaler_param)[:, 0]
df['MonthlyIncome_scaler'] = scaler.fit_transform(df[['Age', 'MonthlyIncome', 'PercentSalaryHike', 'TotalWorkingYears', 'YearsAtCompany', 'YearsInCurrentRole', 
                                            'YearsSinceLastPromotion', 'YearsWithCurrManager', 'DistanceFromHome']], scaler_param)[:, 1]
df['PercentSalaryHike_scaler'] = scaler.fit_transform(df[['Age', 'MonthlyIncome', 'PercentSalaryHike', 'TotalWorkingYears', 'YearsAtCompany', 'YearsInCurrentRole', 
                                            'YearsSinceLastPromotion', 'YearsWithCurrManager', 'DistanceFromHome']], scaler_param)[:, 2]
df['TotalWorkingYears_scaler'] = scaler.fit_transform(df[['Age', 'MonthlyIncome', 'PercentSalaryHike', 'TotalWorkingYears', 'YearsAtCompany', 'YearsInCurrentRole', 
                                            'YearsSinceLastPromotion', 'YearsWithCurrManager', 'DistanceFromHome']], scaler_param)[:, 3]
df['YearsAtCompany_scaler'] = scaler.fit_transform(df[['Age', 'MonthlyIncome', 'PercentSalaryHike', 'TotalWorkingYears', 'YearsAtCompany', 'YearsInCurrentRole', 
                                            'YearsSinceLastPromotion', 'YearsWithCurrManager', 'DistanceFromHome']], scaler_param)[:, 4]
df['YearsInCurrentRole_scaler'] = scaler.fit_transform(df[['Age', 'MonthlyIncome', 'PercentSalaryHike', 'TotalWorkingYears', 'YearsAtCompany', 'YearsInCurrentRole', 
                                            'YearsSinceLastPromotion', 'YearsWithCurrManager', 'DistanceFromHome']], scaler_param)[:, 5]
df['YearsSinceLastPromotion_scaler'] = scaler.fit_transform(df[['Age', 'MonthlyIncome', 'PercentSalaryHike', 'TotalWorkingYears', 'YearsAtCompany', 'YearsInCurrentRole', 
                                            'YearsSinceLastPromotion', 'YearsWithCurrManager', 'DistanceFromHome']], scaler_param)[:, 6]
df['YearsWithCurrManager_scaler'] = scaler.fit_transform(df[['Age', 'MonthlyIncome', 'PercentSalaryHike', 'TotalWorkingYears', 'YearsAtCompany', 'YearsInCurrentRole', 
                                            'YearsSinceLastPromotion', 'YearsWithCurrManager', 'DistanceFromHome']], scaler_param)[:, 7]
df['DistanceFromHome_scaler'] = scaler.fit_transform(df[['Age', 'MonthlyIncome', 'PercentSalaryHike', 'TotalWorkingYears', 'YearsAtCompany', 'YearsInCurrentRole', 
                                            'YearsSinceLastPromotion', 'YearsWithCurrManager', 'DistanceFromHome']], scaler_param)[:, 8]
df

#逻辑回归建模
from sklearn import linear_model
train_df = df.filter(regex='Attrition|Age_.*|DistancenFromHome_.*|BusinessTravel_.*|Department_.*|Education_.*|EducationField_.*|EnvironmentSatisfaction_.*|Gender_.*|JobInvolvement_.*|JobLevel_.*|JobRole_.*|JobSatisfaction_.*|MaritalStatus_.*|MonthlyIncome|NumCompaniesWorked|OverTime_.*|PercentSalaryHike_.*|PerformanceRating_.*|RelationshipSatisfaction_.*|StockOptionLevel_.*|TotalWorkingYears_.*|TrainingTimesLastYear_.*|WorkLifeBalance_.*|YearsInCompany_.*|YearsInCurrentRole_.*|YearsSinceLastPromotion_.*|YearsWithCurrManager_.*')
train_np = train_df.as_matrix()
y = train_np[:, 0]
X = train_np[:, 1:]
clf = linear_model.LogisticRegression(C=1.0, penalty='l1', tol=1e-6)
clf.fit(X, y)

clf

#导入测试数据
test_data = pd.read_csv(r'C:\Users\WENNY CUNXI\Desktop\taitannike\b.csv')
test_data.info()
test_data.describe()

#测试数据特征因子化、数据合并和删减
dummies_BusinessTravel = pd.get_dummies(test_data['BusinessTravel'], prefix='BusinessTravel')
dummies_Department = pd.get_dummies(test_data['Department'], prefix='Department')
dummies_Education = pd.get_dummies(test_data['Education'], prefix='Education')
dummies_EducationField = pd.get_dummies(test_data['EducationField'], prefix='EducationField')
dummies_EnvironmentSatisfaction = pd.get_dummies(test_data['EnvironmentSatisfaction'], prefix='EnvironmentSatisfaction')
dummies_Gender = pd.get_dummies(test_data['Gender'], prefix='Gender')
dummies_JobInvolvement = pd.get_dummies(test_data['JobInvolvement'], prefix='JobInvolvement')
dummies_JobLevel = pd.get_dummies(test_data['JobLevel'], prefix='JobLevel')
dummies_JobRole = pd.get_dummies(test_data['JobRole'], prefix='JobRole')
dummies_JobSatisfaction = pd.get_dummies(test_data['JobSatisfaction'], prefix='JobSatisfaction')
dummies_MaritalStatus = pd.get_dummies(test_data['MaritalStatus'], prefix='MaritalStatus')
dummies_NumCompaniesWorked = pd.get_dummies(test_data['NumCompaniesWorked'], prefix='NumCompaniesWorked')
dummies_OverTime = pd.get_dummies(test_data['OverTime'], prefix='OverTime')
dummies_PerformanceRating = pd.get_dummies(test_data['PerformanceRating'], prefix='PerformanceRating')
dummies_RelationshipSatisfaction = pd.get_dummies(test_data['RelationshipSatisfaction'], prefix='RelationshipSatisfaction')
dummies_StockOptionLevel = pd.get_dummies(test_data['StockOptionLevel'], prefix='StockOptionLevel')
dummies_TrainingTimesLastYear = pd.get_dummies(test_data['TrainingTimesLastYear'], prefix='TrainingTimesLastYear')
dummies_WorkLifeBalance = pd.get_dummies(test_data['WorkLifeBalance'], prefix='WorkLifeBalance')

df = pd.concat([test_data, dummies_BusinessTravel, dummies_Department, dummies_Education, dummies_EducationField, dummies_EnvironmentSatisfaction,
               dummies_Gender, dummies_JobInvolvement, dummies_JobLevel, dummies_JobRole, dummies_JobSatisfaction, dummies_MaritalStatus, dummies_NumCompaniesWorked,
               dummies_OverTime, dummies_PerformanceRating, dummies_RelationshipSatisfaction, dummies_StockOptionLevel, dummies_TrainingTimesLastYear, dummies_WorkLifeBalance],
               axis=1)

df.drop(['BusinessTravel', 'Department', 'Education', 'EducationField', 'EnvironmentSatisfaction', 'Gender', 'JobInvolvement', 'JobLevel',
        'JobRole', 'JobSatisfaction', 'MaritalStatus', 'NumCompaniesWorked', 'OverTime', 'PerformanceRating', 'RelationshipSatisfaction',
        'StockOptionLevel', 'TrainingTimesLastYear', 'WorkLifeBalance'], axis=1, inplace=True)
df

#测试数据数据特征化到[-1, 1]之间
scaler_param = scaler.fit(df[['Age', 'MonthlyIncome', 'PercentSalaryHike', 'TotalWorkingYears', 'YearsAtCompany', 'YearsInCurrentRole', 
                                  'YearsSinceLastPromotion', 'YearsWithCurrManager', 'DistanceFromHome']])
df['Age_scaler'] = scaler.fit_transform(df[['Age', 'MonthlyIncome', 'PercentSalaryHike', 'TotalWorkingYears', 'YearsAtCompany', 'YearsInCurrentRole', 
                                            'YearsSinceLastPromotion', 'YearsWithCurrManager', 'DistanceFromHome']], scaler_param)[:, 0]
df['MonthlyIncome_scaler'] = scaler.fit_transform(df[['Age', 'MonthlyIncome', 'PercentSalaryHike', 'TotalWorkingYears', 'YearsAtCompany', 'YearsInCurrentRole', 
                                            'YearsSinceLastPromotion', 'YearsWithCurrManager', 'DistanceFromHome']], scaler_param)[:, 1]
df['PercentSalaryHike_scaler'] = scaler.fit_transform(df[['Age', 'MonthlyIncome', 'PercentSalaryHike', 'TotalWorkingYears', 'YearsAtCompany', 'YearsInCurrentRole', 
                                            'YearsSinceLastPromotion', 'YearsWithCurrManager', 'DistanceFromHome']], scaler_param)[:, 2]
df['TotalWorkingYears_scaler'] = scaler.fit_transform(df[['Age', 'MonthlyIncome', 'PercentSalaryHike', 'TotalWorkingYears', 'YearsAtCompany', 'YearsInCurrentRole', 
                                            'YearsSinceLastPromotion', 'YearsWithCurrManager', 'DistanceFromHome']], scaler_param)[:, 3]
df['YearsAtCompany_scaler'] = scaler.fit_transform(df[['Age', 'MonthlyIncome', 'PercentSalaryHike', 'TotalWorkingYears', 'YearsAtCompany', 'YearsInCurrentRole', 
                                            'YearsSinceLastPromotion', 'YearsWithCurrManager', 'DistanceFromHome']], scaler_param)[:, 4]
df['YearsInCurrentRole_scaler'] = scaler.fit_transform(df[['Age', 'MonthlyIncome', 'PercentSalaryHike', 'TotalWorkingYears', 'YearsAtCompany', 'YearsInCurrentRole', 
                                            'YearsSinceLastPromotion', 'YearsWithCurrManager', 'DistanceFromHome']], scaler_param)[:, 5]
df['YearsSinceLastPromotion_scaler'] = scaler.fit_transform(df[['Age', 'MonthlyIncome', 'PercentSalaryHike', 'TotalWorkingYears', 'YearsAtCompany', 'YearsInCurrentRole', 
                                            'YearsSinceLastPromotion', 'YearsWithCurrManager', 'DistanceFromHome']], scaler_param)[:, 6]
df['YearsWithCurrManager_scaler'] = scaler.fit_transform(df[['Age', 'MonthlyIncome', 'PercentSalaryHike', 'TotalWorkingYears', 'YearsAtCompany', 'YearsInCurrentRole', 
                                            'YearsSinceLastPromotion', 'YearsWithCurrManager', 'DistanceFromHome']], scaler_param)[:, 7]
df['DistanceFromHome_scaler'] = scaler.fit_transform(df[['Age', 'MonthlyIncome', 'PercentSalaryHike', 'TotalWorkingYears', 'YearsAtCompany', 'YearsInCurrentRole', 
                                            'YearsSinceLastPromotion', 'YearsWithCurrManager', 'DistanceFromHome']], scaler_param)[:, 8]
df

#运用建好的模型对测试数据中员工任职情况进行预测，并导出结果
test_df = df.filter(regex='Attrition|Age_.*|DistancenFromHome_.*|BusinessTravel_.*|Department_.*|Education_.*|EducationField_.*|EnvironmentSatisfaction_.*|Gender_.*|JobInvolvement_.*|JobLevel_.*|JobRole_.*|JobSatisfaction_.*|MaritalStatus_.*|MonthlyIncome|NumCompaniesWorked|OverTime_.*|PercentSalaryHike_.*|PerformanceRating_.*|RelationshipSatisfaction_.*|StockOptionLevel_.*|TotalWorkingYears_.*|TrainingTimesLastYear_.*|WorkLifeBalance_.*|YearsInCompany_.*|YearsInCurrentRole_.*|YearsSinceLastPromotion_.*|YearsWithCurrManager_.*')
predictions = clf.predict(test_df)
result = pd.DataFrame({'EmployeeNumber':test_data['EmployeeNumber'].as_matrix(), 'Arrrition':predictions.astype(np.int64)})
result.to_csv(r'C:\Users\WENNY CUNXI\Desktop\taitannike\ttnkresult.csv', index=False)

