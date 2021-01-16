
import time
import numpy as np
import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt




def getCounter(df, col):
    tmp = df[col].value_counts()
    print ('  --> Col : ', col, ' || Vals : ', list(zip(tmp.keys(), tmp.tolist())))

# This functions reads the .csv file and makes minor changes to it before writing
# it as data_for_student_case_v1.csv. Here, we remove 'Refused' transactions and drop
# values with NA. The CVCResponse codes > 2 are all coded as 3. Creation data is converted
# to datetime format.

def getData(filename):
    df = pd.read_csv(filename);
    print(' - Objects : ', len(df))
    getCounter(df, 'simple_journal')

    df.dropna(inplace=True)
    print(' - Objects (no NA) : ', len(df))
    getCounter(df, 'simple_journal')

    print(' - Dropping Refused')
    df.drop(df[df['simple_journal'] == 'Refused'].index, axis=0, inplace=True)
    getCounter(df, 'simple_journal')

    df.loc[df['cvcresponsecode'] > 2, 'cvcresponsecode'] = 3
    df['Y'] = np.where(df['simple_journal'] == 'Chargeback', 1, 0)
    df['creationdate'] = pd.to_datetime(df['creationdate'])

    # print (df.head())
    df.to_csv('../data/data_for_student_case_v1.csv', index=False)
    print(' - Saved File!')

    return df

# In the following function, rigorous processing of the data is done.
# This involves getting the hours, converting the amount etc. Going thtough
# the code will more or less explain what is going on.

def getProcessedData(df, currency_conv, column_coding=False, identifiers=False):
    if (0):  # printing
        print('txvariantcode : ', df['txvariantcode'].unique())
        print('shopperinteraction : ', df['shopperinteraction'].unique())
        print('accountcode : ', df['accountcode'].unique())
        print('currencycode : ', df['currencycode'].unique())
        print('cvcresponsecode : ', df['cvcresponsecode'].unique())
        print('simple_journal : ', df['simple_journal'].unique())
        print('------------------------------')

    if (1):

        df['creationdate'] = pd.to_datetime(df['creationdate'])
        df['hrs'] = df['creationdate'].dt.hour
        df['amount1'] = df['amount'] / 100.0
        df['convertedAmount'] = df['amount1'].copy()
        for currency in currency_conv.keys():
            # print (' Currency : ', currency, ' || Coversion :', currency_conv[currency])
            idxs = df[df['currencycode'] == currency].index.tolist()
            df.loc[idxs, 'convertedAmount'] = df.loc[idxs, 'amount1'] * currency_conv[currency]

        df['convertedAmount_norm'] = (df[['convertedAmount']] - df[['convertedAmount']].min()) / (
                    df['convertedAmount'].max() - df['convertedAmount'].min())

        df['cardverificationcodesupplied'] = df['cardverificationcodesupplied'].astype("category").cat.codes

    if (column_coding == True):  # convert categorical to codes
        df['issuercountrycode'] = df['issuercountrycode'].astype("category").cat.codes
        df['txvariantcode'] = df['txvariantcode'].astype("category").cat.codes
        df['currencycode'] = df['currencycode'].astype("category").cat.codes
        df['shoppercountrycode'] = df['shoppercountrycode'].astype("category").cat.codes
        df['shopperinteraction'] = df['shopperinteraction'].astype("category").cat.codes
        df['accountcode'] = df['accountcode'].astype("category").cat.codes
        df['cardverificationcodesupplied'] = df['cardverificationcodesupplied'].astype("category").cat.codes

    if (1):  # dropping
        df.drop('txid', axis=1, inplace=True)
        df.drop('bookingdate', axis=1, inplace=True)
        df.drop('bin', axis=1, inplace=True)
        df.drop('simple_journal', axis=1, inplace=True)
        # df.drop('amount', axis=1, inplace=True)
        if (identifiers == False):
            df.drop('mail_id', axis=1, inplace=True)
            df.drop('ip_id', axis=1, inplace=True)
            df.drop('card_id', axis=1, inplace=True)

    if (1):  # printing
        print(' - issuercountrycode (unique count)  : ', len(df['issuercountrycode'].unique()))
        print(' - shoppercountrycode (unique count) : ', len(df['shoppercountrycode'].unique()))

    getCounter(df, 'Y')

    cols = df.columns.tolist()
    cols.remove('Y')
    df = df[cols + ['Y']]

    if (column_coding == True and identifiers == False):
        df.to_csv('../data/data_for_student_case_v2.csv', index=False)
    elif (column_coding == False and identifiers == False):
        df.to_csv('../data/data_for_student_case_v3.csv', index=False)
    elif (identifiers == True):
        df.to_csv('../data/data_for_student_case_v4.csv', index=False)

    return df

# In the following function, we attempt one hot encoding for certain features to see
# if that helps in the classification tasks.

def encoding(df, type_encoding):
    print(' - Original Column Count : ', len(df.columns))

    if (type_encoding == 'onehot'):
        cols_onehot = ['txvariantcode', 'currencycode', 'shopperinteraction', 'cvcresponsecode', 'accountcode']
        for col in cols_onehot:
            df[col] = pd.Categorical(df[col])
            df1 = pd.get_dummies(df[col], prefix=col)
            print(' - Col :', col, ' || Extra cols added : ', len(df1.columns))
            df = pd.concat([df, df1], axis=1)
            df.drop(col, axis=1, inplace=True)
    elif (type_encoding == 'codes'):
        cols_code = ['txvariantcode', 'currencycode', 'shopperinteraction', 'cvcresponsecode', 'accountcode']
        for col in cols_code:
            df[col] = df[col].astype('category').cat.codes

    if (1):
        countries = ['MX', 'AU', 'GB', 'SE', 'NZ', 'BR', 'FI']

        for col in ['issuercountrycode', 'shoppercountrycode']:
            for countrycode in countries:
                df[col + '_' + countrycode] = np.where(df[col] == countrycode, 1, 0)

            countries_unique = np.array(df[col].unique().tolist())
            countries_not = np.delete(countries_unique, np.where(np.isin(countries, countries_unique)))
            df[col + '_none'] = np.where(np.isin(df[col], countries_not), 1, 0)

            df.drop(col, axis=1, inplace=True)

    print(' - Final Column Count : ', len(df.columns))

    if (1):
        df.drop(['amount', 'amount1', 'convertedAmount'], axis=1, inplace=True)

    cols = df.columns.tolist()
    cols.remove('Y')
    df = df[cols + ['Y']]

    if (type_encoding == 'onehot'):
        df.to_csv('../data/data_for_student_case_v5.csv', index=False)
    elif (type_encoding == 'codes'):
        df.to_csv('../data/data_for_student_case_v6.csv', index=False)

    return df

