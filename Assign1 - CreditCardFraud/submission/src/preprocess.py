
import time
import numpy as np
import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt



# ----------------- HELPER FUNCTIONS -------------------------

def getCounter(df, col):
    tmp = df[col].value_counts()
    print ('  --> Col : ', col, ' || Vals : ', list(zip(tmp.keys(), tmp.tolist())))

def getData1(filename):

    # Step1 - Get Data
    print (' - 1. Get Data')
    df = pd.read_csv(filename);

    # Step2 - Drop NaN values
    print (' - 2. Drop NaN vals')
    df.dropna(inplace=True)

    # Step3 - Modified Columns
    print (' - 3. Modify Columns')
    df.drop(df[df['simple_journal'] == 'Refused'].index, axis=0, inplace=True)
    df.loc[df['cvcresponsecode'] > 2, 'cvcresponsecode'] = 3
    df['Y']                            = np.where(df['simple_journal'] == 'Chargeback', 1, 0)
    df['cardverificationcodesupplied'] = df['cardverificationcodesupplied'].astype("category").cat.codes
    df['creationdate']                 = pd.to_datetime(df['creationdate'])


    # Step4 - Created Columns
    print (' - 4. Creating new columns')
    df['hrs']             = df['creationdate'].dt.hour
    df['amount1']         = df['amount'] / 100.0
    df['convertedAmount'] = df['amount1'].copy()
    currency_conv         = {'SEK': 0.09703, 'MXN': 0.04358, 'AUD': 0.63161, 'NZD': 0.58377, 'GBP': 1.13355}
    for currency in currency_conv.keys():
        idxs = df[df['currencycode'] == currency].index.tolist()
        df.loc[idxs, 'convertedAmount'] = df.loc[idxs, 'amount1'] * currency_conv[currency]
    df['convertedAmount_norm'] = (df[['convertedAmount']] - df[['convertedAmount']].min()) / (
            df['convertedAmount'].max() - df['convertedAmount'].min())

    # Step5- Drop Rows/Columns
    print (' - 5. Dropping Columns')
    df.drop('txid', axis=1, inplace=True)
    df.drop('bookingdate', axis=1, inplace=True)
    df.drop('bin', axis=1, inplace=True)
    df.drop('simple_journal', axis=1, inplace=True)
    df.drop('mail_id', axis=1, inplace=True)
    df.drop('ip_id', axis=1, inplace=True)
    df.drop('card_id', axis=1, inplace=True)
    df.drop(['amount', 'amount1'],axis=1, inplace=True)

    print (' - 6. Rearranging columns as (X|Y)')
    cols = df.columns.tolist()
    cols.remove('Y')
    df = df[cols + ['Y']]

    print (' - 7. Saving to .csv file')
    df.to_csv('data/data_for_student_case_preprocessed.csv', index=False)

    return df

def getData2(df, type_encoding):
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

    df.drop('hrs', axis=1, inplace=True)
    df.drop('convertedAmount', axis=1, inplace=True)
    df.drop('creationdate', axis=1, inplace=True)

    print(' - Final Column Count : ', len(df.columns))

    cols = df.columns.tolist()
    cols.remove('Y')
    df = df[cols + ['Y']]

    if (type_encoding == 'onehot'):
        df.to_csv('data/data_for_student_case_preprocessed_onehot.csv', index=False)
    elif (type_encoding == 'codes'):
        df.to_csv('data/data_for_student_case_preprocessed_codes.csv', index=False)

    return df

"""
Rough Code Snippets
for each in zip(Y_tests, Y_tests_preds_probabs):
    if each[0] == 1:
        print (each[0], round(each[1],4))
"""
