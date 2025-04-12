from django.http import HttpResponse
from django.shortcuts import render
import pickle
import warnings
import yfinance as yf
import pandas as pd
import numpy as np
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

warnings.filterwarnings('ignore')


def grab_price_data(stock):
    start = '2022-01-01'
    end = '2024-05-07'
    print("works till here")
    data = yf.download(stock, start, end, auto_adjust=False)
    data.reset_index(inplace=True)
    print(f"Data shape: {data.shape}")
    print(data)
    return data


def lstm_pred(data):
    data_train = pd.DataFrame(data.Close[0: int(len(data)*0.80)])
    data_test = pd.DataFrame(data.Close[int(len(data)*0.80): len(data)])
    scaler = MinMaxScaler(feature_range=(0, 1))
    pas_100_days = data_train.tail(100)
    data_test = pd.concat([pas_100_days, data_test], ignore_index=True)
    data_test_scale = scaler.fit_transform(data_test)
    x, y = [], []

    for i in range(100, data_test_scale.shape[0]):
        x.append(data_test_scale[i-100:i])
        y.append(data_test_scale[i, 0])
    x, y = np.array(x), np.array(y)
    model = load_model("D:\\maj_proj\\stock_price_prediction\\StockPrice\\stock\\lstm_pred.h5")
    y_predict = model.predict(x)
    scale = 1 / scaler.scale_
    y_predict = y_predict * scale
    y = y * scale
    last_ypred = y_predict[99][0]
    last_yorig = y[99]
    return 1 if last_yorig > last_ypred else -1


def knn_pred(data):
    with open('D:\\maj_proj\\stock_price_prediction\\StockPrice\\stock\\knn_price_pred_model.pkl', 'rb') as f:
        model = pickle.load(f)
        data['open-close'] = data['Open'] - data['Close']
        data['high-low'] = data['High'] - data['Low']
        data = data.dropna()
        last_row = data.tail(1)
        X = last_row[['open-close', 'high-low']]
        predicted = model.predict(X)
        return predicted[0]


# ✅ FIXED OBV FUNCTION (outside now)
def obv(data):
    volume = data['Volume'].values
    close_diff = data['Close'].diff().fillna(0).values
    obv_values = []
    prev_obv = 0

    for change, vol in zip(close_diff, volume):
        if change > 0:
            current_obv = prev_obv + vol
        elif change < 0:
            current_obv = prev_obv - vol
        else:
            current_obv = prev_obv
        obv_values.append(current_obv)
        prev_obv = current_obv

    return pd.Series(obv_values, index=data.index)


def rndfr_pred(price_data):
    price_data.drop(columns=['Adj Close'], inplace=True)
    price_data = price_data[['Date', 'Close', 'High', 'Low', 'Open', 'Volume']]
    price_data.sort_values(by=['Date'], inplace=True)
    price_data['change_in_price'] = price_data['Close'].diff()
    n = 14

    up_df, down_df = price_data[['change_in_price']].copy(), price_data[['change_in_price']].copy()
    up_df.loc[price_data['change_in_price'] < 0, 'change_in_price'] = 0
    down_df.loc[price_data['change_in_price'] > 0, 'change_in_price'] = 0
    down_df['change_in_price'] = down_df['change_in_price'].abs()

    ewma_up = up_df['change_in_price'].ewm(span=n).mean()
    ewma_down = down_df['change_in_price'].ewm(span=n).mean()
    relative_strength = ewma_up / ewma_down
    relative_strength_index = 100.0 - (100.0 / (1.0 + relative_strength))

    price_data['down_days'] = down_df['change_in_price']
    price_data['up_days'] = up_df['change_in_price']
    price_data['RSI'] = relative_strength_index

    low_14 = price_data['Low'].rolling(window=n).min()
    high_14 = price_data['High'].rolling(window=n).max()
    k_percent = 100 * ((price_data['Close'] - low_14) / (high_14 - low_14))
    price_data['low_14'] = low_14
    price_data['high_14'] = high_14
    price_data['k_percent'] = k_percent

    r_percent = ((high_14 - price_data['Close']) / (high_14 - low_14)) * -100
    price_data['r_percent'] = r_percent

    ema_26 = price_data['Close'].ewm(span=26).mean()
    ema_12 = price_data['Close'].ewm(span=12).mean()
    macd = ema_12 - ema_26
    ema_9_macd = macd.ewm(span=9).mean()
    price_data['MACD'] = macd
    price_data['MACD_EMA'] = ema_9_macd

    price_data['Price_Rate_Of_Change'] = price_data['Close'].pct_change(periods=9)

    # ✅ Apply fixed OBV function
    price_data['On Balance Volume'] = obv(price_data)

    X_Cols = price_data[['RSI', 'k_percent', 'r_percent', 'Price_Rate_Of_Change', 'MACD', 'On Balance Volume']]
    X = X_Cols.tail(1)

    with open("D:\\maj_proj\\stock_price_prediction\\StockPrice\\stock\\rand_frst.pkl", 'rb') as f:
        model = pickle.load(f)
        predicted = model.predict(X)
        return int(predicted[0])


def my_view(request):
    out = None
    if request.method == 'POST':
        stock = request.POST.get('stock')

        if not stock:
            return HttpResponse('Please provide a stock symbol.')

        data = grab_price_data(stock)

        if data is None:
            return HttpResponse('Failed to retrieve data for the given stock symbol.')

        lstm = lstm_pred(data)
        knn = knn_pred(data)
        rdm = rndfr_pred(data)

        if lstm == 1 and knn == 1 and rdm == 1:
            out = 'STRONG BUY'
        elif lstm == -1 and knn == 1 and rdm == 1:
            out = 'BUY'
        elif lstm == 1 and knn == -1 and rdm == 1:
            out = 'BUY'
        elif lstm == 1 and knn == 1 and rdm == -1:
            out = 'BUY'
        elif lstm == -1 and knn == -1 and rdm == 1:
            out = 'SELL'
        elif lstm == 1 and knn == -1 and rdm == -1:
            out = 'SELL'
        elif lstm == -1 and knn == 1 and rdm == -1:
            out = 'SELL'
        elif lstm == -1 and knn == -1 and rdm == -1:
            out = 'STRONG SELL'

        context = {
            'lstm': lstm,
            'knn': knn,
            'rdm': rdm,
            'out': out,
        }
        return render(request, 'output.html', context)
    else:
        return HttpResponse('not submitted')


def home(request):
    my_view(request)
    return render(request, 'home.html')
