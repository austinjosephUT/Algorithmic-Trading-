#initial code to be working effectively in terms of gathering the S&P500 data
#I have commented throughout the code for clarification when coming back

import bs4 as bs #for scraping data out of public financial api's
import datetime as dt #for using the date
import os #managing directories
import pandas as pd #data analysis
import pandas_datareader.data as web
import pickle
import requests
import numpy as np

import matplotlib.pyplot as plt
import matplotlib import style

#pickle helps save the lists

from sklearn import svm, neighbors, cross_validation
from sklearn.ensemble import VotingClassifier, RandomForestClassifier



def save_sp500_tickets():
    resp = requests.get('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
    soup = bs.BeautifulSoup(resp.text, "lxml")
    table = soup.find('table', {'class':'wikitable sortable'})
    tickers = []
    for row in table.findAll('tr')[1:]:
        ticker = row.findAll('td')[0].text
        tickers.append(ticker)
    with open("sp500tickers.pickle", "wb") as f:
        pickle.dump(tickers, f)

    tickers.__delitem__(0)
    print(tickers)
    return tickers


save_sp500_tickets()

def get_data_goog(reload_sp500=False):
    if reload_sp500:
        tickers = save_sp500_tickets()
    else:
        with open("sp500tickers.pickle", "rb") as f:
            tickers = pickle.load(f)

    if not os.path.exists('stock_dfs'):
        os.makedirs('stock_dfs')

    start = dt.datetime(2000,1,1)
    end = dt.datetime(2016,12,31)

    for ticker in tickers:
        print(ticker)
        if not os.path.exists('stock_dfs/{}.csv'.format(ticker)):
            df = web.DataReader(ticker, 'yahoo', start, end)
            df.to_csv('stock_dfs/{}.csv'.format(ticker))
        else:
            print('Already have {}'.format(ticker))


def compile_data():
    with open("sp500tickers.pickle", "rb") as f:
        tickers = pickle.load(f)

    main_df = pd.DataFrame()

    for count, ticker in enumerate(tickers):
        df = pd.read_csv('stock_dfs/{}.csv'.format(ticker))
        df.set_index('Date', inplace=True)

        df.rename(columns = {'Adj Close':ticker}, inplace=True)
        df.drop(['Open','High','Low','Volume'],1,inplace=True)

        if main_df.empty:
            main_df = df
        else:
            main_df = main_df.join(df, how='outer')

        if count % 10 == 0:
            print(count)

    print(main_df.head())
    main_df.to_csv('sp500_joined_closes.csv')


get_data_goog(reload_sp500=True)
compile_data()








#This code may not be working effectively in terms of gathering the S&P500 data becasuse the tutorial was a bit outdated
#make sure to look into the quandl api subscription services and etc

style.use('ggplot')

def visualize_data():
    df = pd.read_csv('sp500_joined_closes.csv')
    df['AAPL'].plot()
    plt.show()
    df_corr = df.corr()
    print(df.corr().head())

    data = df_corr.values
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)

    heatmap = ax.pcolor(data, cmap=plt.cm.RdY1Gn)
    fig.colorbar(heatmap)
    ax.set_xticks(np.arrange(data.shape[0]) +0.5, minor=False)
    ax.set_yticks(np.arrange(data.shape[1]) +0.5, minor=False)
    ax.invert_yaxis()
    ax.xaxis.tick_top()

    column_labels = df_corr.columns
    row_labels = df_corr.index

    ax.set_xticks(column_labels)
    ax.set_yticks(row_labels)
    plt.xticks(rotation=90)
    heatmap.set_clim(-1,1)
    plt.tight_layout()
    plt.show()


visualize_data()
#run on a good processer because the amount of data is HUUUUGEE!


def process_data_for_labels(ticker):
    hm_days = 7
    df = pdf.read_csv('sp500_joined_closes.csv',index_col=0)
    tickers = df.columns.values.tolist()
    df.fillna(0,inplace=True)

    for i in range(1,hm_days+1):
        df['{}_{}d'.format(ticker,i)] = (df[ticker].shift(-i) - df[ticker]) / df[ticker]

    df.fillna(0,inplace=True)
    return tickers,df

process_data_for_labels('XOM')


def buy_sell_hold(*args):
    cols = [c for c in args]
    requirement = 0.02
    for col in cols:
        if col>requirement:
            return 1
        if col<-requirement:
            return -1
    return 0

def extract_feature_sets(ticker):
    tickers,df = process_data_for_labels(ticker)

    df['{}_target'.format(ticker)] = list(map(buy_sell_hold(),
                                              df['{}_1d']))








def do_ml(ticker):
    X, y, df = extract_feature_sets(ticker)
    X_train, X_test, y_train, y_test = cross_validation.train_test_split(X,y, )

    clf = neighbors.KNeighborsClassifier()

    clf = Bot



    clf.fit(X_train,y_train)
    confidence = clf.score(X_test, y_test)
    print('Accuracy', confidence)
    predictions == clf.predict(X_test)
    print('Predicted spread:', Counter(predictions))
    return confidence


do_ml('BAC')












