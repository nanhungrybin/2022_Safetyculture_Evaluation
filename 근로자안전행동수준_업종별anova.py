#!/usr/bin/env python
# coding: utf-8

# In[49]:


import seaborn as sns 
import matplotlib.pyplot as plt
import pandas as pd
# Raw Data
raw = pd.read_csv("C:/data_analysis/9th_siltae_public_new_211102.csv")

# 결측치 열 제거
raw = raw.dropna(axis = 1)

# 원하는 열만 추출
new = raw[["SQ2","Q14_1","Q15_1","Q21_1_1","Q21_1_2","Q22","Q25_1","Q25_2","Q25_3","Q25_4","Q25_5","Q25_6","Q25_7","Q25_8","Q29_5",
           "Q29_6","Q29_7","Q39","Q40_1","Q40_2","Q40_3","Q40_4","Q40_5","Q40_6",
           "Q40_7","Q26_1","Q26_2","Q26_3","Q26_4"]]

# 이상치 값 대체 - 9998 to 3(보통)
new = new.replace({'Q29_5': 9998}, {'Q29_5': 3})
new = new.replace({'Q29_6': 9998}, {'Q29_6': 3})
#Q14_1과 Q15_1을 전체 근로자수 (Q2_1D3)으로 나누기
new["Q14_1"] =   new["Q14_1"] / raw["Q2_1D3"]
new["Q15_1"] =   new["Q15_1"] / raw["Q2_1D3"]

# 종속변수 평균값 대체
change = new[['Q26_1','Q26_2','Q26_3','Q26_4']] #대체 할 4개의 종속변수
m1 = change.mean(axis=1)
new.loc[:,'mean']=m1
new = new.drop(["Q26_1","Q26_2","Q26_3","Q26_4"], axis = 1)

#더미변수 지정
#pd.get_dummies(new, columns = ["Q21_1_1","Q21_1_2","Q22"])
new = new.replace({'Q21_1_1': 2}, {'Q21_1_1': 0})

new = new.replace({'Q21_1_2': 2}, {'Q21_1_2': 0})

new = new.replace({'Q22': 2}, {'Q22': 0})

# inf값 대체
import numpy as np

new = new.replace(np.inf, 0)


# 독립변수, 종속변수 이름 대체
# 독립변수 : 기존 to Q1 ~ Q24
# 종속변수 : 기존 mean to worker safety level

new.columns #기존 열 이름 확인

for i in range(1,26,1):
    print("Q%d" %i, end = " ")
    
new.columns = ["group","Q1","Q2","Q3","Q4","Q5","Q6","Q7","Q8","Q9"
               ,"Q10","Q11","Q12","Q13","Q14","Q15","Q16"
               ,"Q17","Q18","Q19","Q20","Q21","Q22","Q23","Q24",'worker_safety_level']

new


plt.figure(figsize=(8,4))

sns.distplot(new.loc[ new['group']== 1]['worker_safety_level'], hist=False)
sns.distplot(new.loc[ new['group']== 2]['worker_safety_level'], hist=False)
sns.distplot(new.loc[ new['group']== 3]['worker_safety_level'], hist=False)
plt.title('difference between group')
plt.show()

sns.set_style("whitegrid")
plt.figure(figsize=(6,5))
sns.boxplot( data=new, x='group', y='worker_safety_level')
plt.title('difference between group')
plt.show()

sns.scatterplot(x='group', y='worker_safety_level', data=new)
plt.show()

group1 = new[new['group']==1]['worker_safety_level']
group2 = new[new['group']==2]['worker_safety_level']
group3 = new[new['group']==3]['worker_safety_level']

print('group1 평균 근로자 안전행동수준:', group1.mean(), 'group1 표준 편차:', group1.std())
print('group2 평균 근로자 안전행동수준:', group2.mean(), 'group2 표준 편차:', group2.std())
print('group3 평균 근로자 안전행동수준:', group3.mean(), 'group3 표준 편차:', group3.std())


# In[48]:


from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
from scipy import stats
import pandas as pd

#일원분산분석(One-way ANOVA)
#정규성 검정
print(stats.shapiro(new.worker_safety_level[new.group==1]))
print(stats.shapiro(new.worker_safety_level[new.group==2]))
print(stats.shapiro(new.worker_safety_level[new.group==3]))

#3개 그룹 모두 p < .05이므로 각 집단의 자료가 정규성을 벗어남

#등분산성 검정
stats.levene( new.worker_safety_level[new.group==1], 
             new.worker_safety_level[new.group==2],
             new.worker_safety_level[new.group==3] )

#p < .05이므로 각 집단의 자료가 등분산성을 갖지 않음

#비모수분석
stats.kruskal(new.loc[new.group ==1, "worker_safety_level"],
	new.loc[new.group ==2, "worker_safety_level"],
	new.loc[new.group ==3, "worker_safety_level"])

#사후분석 진행

from statsmodels.sandbox.stats.multicomp import MultiComparison
comp = MultiComparison(new.worker_safety_level, new.group)

#본페로니
import scipy
result = comp.allpairtest(scipy.stats.ttest_ind, method='bonf')
print(result[0])

#튜키의 HSD
from statsmodels.stats.multicomp import pairwise_tukeyhsd

hsd = pairwise_tukeyhsd(new['worker_safety_level'], new['group'], alpha=0.05)
print(hsd.summary())


# In[ ]:




