<<<<<<< HEAD

import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

def plot1(df):
    g = sns.FacetGrid(df, col="Y", size=8, hue="Y")
    g.map(sns.distplot,"convertedAmount", kde=True)
    g.set_titles('Fraud transactions:{col_name}')

def plot2(df):
    sns.factorplot(data=df, x="shopperinteraction", y="convertedAmount", hue="Y", kind='bar')
    plt.show()
    f = sns.factorplot(data=df, x="currencycode", y="convertedAmount", hue="Y", kind='bar', col='shopperinteraction')
    f.set_ylabels('Average convertedAmount (USD)', fontsize=12)
    plt.show()

def plot3(df):
    df['creation_hr'] = df.creationdate.dt.hour
    sns.set(font_scale=1.5)
    #f, axarr = plt.subplots(1,figsize = (10,5))
    f = sns.factorplot(data=df, x="creation_hr", y="convertedAmount", hue="Y", kind='point', size=10, aspect = 2)
    plt.xticks(rotation=90)
    f.set_ylabels('Average convertedAmount', fontsize=20)
    f.set_xlabels('creation_hr', fontsize=20)


def plot4(df):
    df = df.groupby(df['Y'])
    df.count()

    f, axarr = plt.subplots(1, 2)

    df.get_group(1)['txvariantcode'].value_counts().plot(kind='pie', ax=axarr[0], fontsize = 10)
    fig = plt.gcf()
    fig.set_size_inches(16, 10)

    df.get_group(0)['txvariantcode'].value_counts().plot(kind='pie', ax=axarr[1])
    fig = plt.gcf()
    fig.set_size_inches(16, 10)
=======

import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

def plot1(df):
    g = sns.FacetGrid(df, col="Y", size=8, hue="Y")
    g.map(sns.distplot,"convertedAmount", kde=True)
    g.set_titles('Fraud transactions:{col_name}')

def plot2(df):
    sns.factorplot(data=df, x="shopperinteraction", y="convertedAmount", hue="Y", kind='bar')
    plt.show()
    f = sns.factorplot(data=df, x="currencycode", y="convertedAmount", hue="Y", kind='bar', col='shopperinteraction')
    f.set_ylabels('Average convertedAmount (USD)', fontsize=12)
    plt.show()

def plot3(df):
    df['creation_date'] = pd.to_datetime(df['creationdate'], format='%Y-%m-%d %H:%M:%S')
    df['creation_month'] = df.creation_date.dt.month
    df['creation_weekday'] = df.creation_date.dt.weekday
    df['creation_hr'] = df.creation_date.dt.hour
    df['creation_day'] = df.creation_date.dt.date
    sns.factorplot(data=df, x="creation_hr", y="convertedAmount", hue="Y", kind='point', size=20)
    plt.xticks(rotation=90)


def plot4(df):
    df = df.groupby(df['Y'])
    df.count()

    f, axarr = plt.subplots(1, 2)

    df.get_group(1)['txvariantcode'].value_counts().plot(kind='pie', ax=axarr[0])
    fig = plt.gcf()
    fig.set_size_inches(16, 10)

    df.get_group(0)['txvariantcode'].value_counts().plot(kind='pie', ax=axarr[1])
    fig = plt.gcf()
    fig.set_size_inches(16, 10)
>>>>>>> c245a7c9ce67639dee047a7c31592bf1700026af
