
import seaborn as sns
import matplotlib.pyplot as plt

def plot1(df):
    g = sns.FacetGrid(df, col="Y", size=8, hue="Y")
    g.map(sns.distplot,"convertedAmount", kde=True)
    g.set_titles('Fraud transactions:{col_name}')



def plot2(df):
    sns.factorplot(data=df, x="shopperinteraction", y="convertedAmount", hue="Y", kind='bar')

def plot3(df):
    f = sns.factorplot(data=df, x="currencycode", y="convertedAmount", hue="Y", kind='bar', col='shopperinteraction')
    f.set_ylabels('Average convertedAmount (USD)', fontsize=12)

def plot4(df):
    sns.factorplot(data=df, x="convertedAmount", y="currencycode", hue="Y", kind='box', size=8, aspect=2)

def plot5(df):
    sns.factorplot(data=df, x="Y", y="convertedAmount", hue="cvcresponsecode")

def plot6(df):
    sns.factorplot(data=df, x="creation_hr", y="convertedAmount", hue="Y", kind='point', size=20)
    plt.xticks(rotation=90)

def plot7(df):
    sns.factorplot(data=df, x="creation_day", y="convertedAmount", hue="Y", size=20, kind="bar")
    plt.xticks(rotation=90)

def plot8(df):
    plt.plot_date(x=df[(df['Y'] == 1)]['creation_day'], y=df[(df['Y'] == 1)]['convertedAmount'], color='red')
    fig = plt.gcf()
    fig.set_size_inches(18.5, 10.5)

def plot9(df):
    plt.plot_date(x=df[(df['Y'] == 0)]['creation_day'], y=df[(df['Y'] == 0)]['convertedAmount'])
    fig = plt.gcf()
    fig.set_size_inches(18.5, 10.5)

def plot10(df):
    df.get_group(1)['txvariantcode'].value_counts().plot(kind='pie')
    fig = plt.gcf()
    fig.set_size_inches(10, 10)

def plot11(df):
    df.get_group(0)['txvariantcode'].value_counts().plot(kind='pie')
    fig = plt.gcf()
    fig.set_size_inches(10, 10)
