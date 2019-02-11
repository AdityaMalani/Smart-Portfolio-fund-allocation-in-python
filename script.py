import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as spo

def symbol_to_path(symbol, base_dir="data"):
        return os.path.join(base_dir, "{}.csv".format(str(symbol)))

def plot_data(df_data,new_df):
    ax = df_data.plot(title="Portfolio comparission", fontsize=10,label='Equal allocation')
    bx = new_df.plot(ax=ax,label='Smart Allocation')
    plt.legend(loc='upper left')
    ax.set_xlabel("Date")
    ax.set_ylabel("Portfolio Value")
    plt.show()

def getDataFrame(symbols,dates):
        df_final = pd.DataFrame(index=dates)
        for symbol in symbols:
                file_path = symbol_to_path(symbol)
                df_temp = pd.read_csv(file_path, parse_dates=True, index_col="Date",usecols=["Date", "Adj Close"], na_values=["nan"])
                df_temp = df_temp.rename(columns={"Adj Close": symbol})
                df_final = df_final.join(df_temp)
        return df_final

def error_func(X,df):
        df = df*X
        portfolio = (df.sum(axis=1))*10000
        portfolio_dr = (portfolio/portfolio.shift(1))-1
        portfolio_dr.iloc[0]=0
        s = portfolio_dr.mean()/portfolio_dr.std()
        return s*(-1)

def eq_constraint(X):
        return np.sum(X)-1
        

symbols = ['RELIANCE.NS','SBIN.NS','TATAMOTORS.NS','VEDL.NS']
start_date = "2018-02-10"
end_date = "2019-02-10"
X = np.array([0.25,0.25,0.25,0.25],dtype='float16')
dates = pd.date_range(start_date, end_date)
df = getDataFrame(symbols,dates)
df.fillna(method='ffill',inplace=True)
df.fillna(method='bfill',inplace=True)
df = df/df.ix[0]
df1 = df*X
portfolio = (df1.sum(axis=1))*10000
constraints = {'type': 'eq', 'fun': eq_constraint}
result = spo.minimize(error_func,X,args=(df),method='SLSQP',bounds=[(0,1),(0,1),(0,1),(0,1)],constraints=constraints)
print(result.x)
print(np.sum(result.x))
df2 = df*result.x
portfolio1 = (df2.sum(axis=1))*10000
plot_data(portfolio,portfolio1)

