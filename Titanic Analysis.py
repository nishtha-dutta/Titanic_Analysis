#!/usr/bin/env python
# coding: utf-8

import pandas as pd
from pandas import Series, DataFrame

titanic_df = pd.read_csv('train.csv')
titanic_df.head()

'''
Here, 
1) PassengerId: unique key for each passenger
2) Survived: 0 - denotes that the passenger didn't servived, 1 - denotes that the passenger had servived
3) Pclass: Class for the passengers on which they are travelling
4) Name: name of the passenger
5) Sex: Gender of the passenger Whether they are male or female
6) Age: Age of the passenger
7) SibSp: whether the passenger has a sbling onboard
8) Parch: Whether the passenger has their parent or children onboard
9) Ticket: Ticket number of the passenger
10) Fare: Amount paid by the passenger to get the Ticket
11) Cabin: Deck number of the passenger where they are staying
12) Embarked: S, C, Q stands for cities where C = Cherbourg, Q = Queenstown, S = Southampton

'''

titanic_df.info()

'''

All data analysis begin with trying to answer questions. 
Now that we know what column category data we have let's think of some questions or insights we would like to obtain from the data.
So, here's a list of questions we'll try to answer using data analysis!

First some basic questions:
1) Who were the passengers on the Titanic?(Age, Gender, Class)
2) What deck were the passengers come from?
3) Where did the passengers come from?
4) Who was alone and who was with family?

Then we'll dig deeper, with a broader question:
5) What factors helped someone survive the sinking?

So, let's start with the first question:
Who were the passengers on the Tatanic?

'''

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

sns.countplot('Sex',data=titanic_df)

sns.countplot('Pclass',data=titanic_df,hue='Sex')

def male_female_child(passenger):
    age,sex = passenger
    if age < 16:
        return 'Child'
    else:
        return sex
    
titanic_df['person'] = titanic_df[['Age','Sex']].apply(male_female_child,axis=1)

sns.countplot('Pclass',data=titanic_df,hue='person')

titanic_df['Age'].hist(bins=70)

titanic_df['Age'].mean()

titanic_df['person'].value_counts()

fig = sns.FacetGrid(titanic_df,hue='Sex',aspect=4)
fig.map(sns.kdeplot,'Age',shade=True)

oldest = titanic_df['Age'].max()
fig.set(xlim=(0,oldest))
fig.add_legend()

fig = sns.FacetGrid(titanic_df,hue='person',aspect=4)
fig.map(sns.kdeplot,'Age',shade=True)

oldest = titanic_df['Age'].max()
fig.set(xlim=(0,oldest))
fig.add_legend()

fig = sns.FacetGrid(titanic_df,hue='Pclass',aspect=4)
fig.map(sns.kdeplot,'Age',shade=True)

oldest = titanic_df['Age'].max()
fig.set(xlim=(0,oldest))
fig.add_legend()

'''
Now, Let's move to the second question
What deck were the passengers come from?
'''

titanic_df.head()

'''
Cabin has the null value so we need to drop them before moving forward
'''

deck = titanic_df['Cabin'].dropna()
deck.head()

levels = []

for level in deck:
    levels.append(level[0])

cabin_df = DataFrame(levels)
cabin_df.columns = ['cabin']
sns.countplot('cabin',data=cabin_df,palette='winter_d')

cabin_df = cabin_df[cabin_df != 'T']

sns.countplot('cabin',data=cabin_df,palette='summer')

'''
Now, Let's move to the Third question
Where did the passengers come from?
'''

titanic_df.head()

sns.countplot('Embarked',data=titanic_df,hue='Pclass')

'''
Now, Let's move to the Forth question
Who was alone and who was with family?
'''

titanic_df.head()

titanic_df['alone'] = titanic_df.SibSp + titanic_df.Parch
titanic_df['alone']

'''
If any passenger has no sbling, parent and child on board then that means they are travelling alone
'''

titanic_df['alone'].loc[titanic_df['alone'] >0] = 'with family'
titanic_df['alone'].loc[titanic_df['alone'] ==0] = 'Alone'

titanic_df.head()

sns.countplot('alone',data=titanic_df,palette='summer')

'''
Now the final question
What factors helped someone survive the sinking?
'''

titanic_df['Survivor'] = titanic_df.Survived.map({0:'No',1:'Yes'})
titanic_df.head()

sns.countplot('Survivor',data=titanic_df,palette='spring')

sns.factorplot('Pclass','Survived',data=titanic_df,hue='person')

sns.lmplot('Age','Survived',data=titanic_df)

generations = [10,20,40,60,80]

sns.lmplot('Age','Survived',data=titanic_df,hue='Pclass',x_bins=generations)

sns.lmplot('Age','Survived',data=titanic_df,hue='Sex',x_bins=generations)

sns.lmplot('Age','Survived',data=titanic_df,hue='alone',x_bins=generations)