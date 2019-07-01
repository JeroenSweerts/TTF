import numpy as np
import pandas as pd
from os import listdir
import random
random.seed()   
import sklearn.mixture as mix
import talib
import matplotlib.pyplot as plt
from matplotlib import cm
import time
import seaborn as sns
from scipy.stats import pearsonr
from hmmlearn import hmm
np.random.seed(42)
from sklearn.externals import joblib

def getX(ticker):
    try:
        import eikon as ek
        ek.set_app_id('604ea22425e048d39af0ef760ec9c64ebbe1fe68')
        startdate='1000-01-20T15:04:05'
        current_date=time.strftime("%Y-%m-%d")

        TTF=ek.get_timeseries(ticker,start_date=startdate,end_date=current_date)
    except:
        TTF=pd.read_csv('data.csv',decimal=',',delimiter=';')
    #     display(TTF)

    TTF = TTF.rename(columns={'CLOSE':'TTF_CLOSE'})
    TTF = TTF.dropna(subset=['TTF_CLOSE'])
    TTF = TTF.rename(columns={'HIGH':'TTF_HIGH'})
    TTF = TTF.dropna(subset=['TTF_HIGH'])
    TTF = TTF.rename(columns={'LOW':'TTF_LOW'})
    TTF = TTF.dropna(subset=['TTF_LOW'])
    TTF = TTF.rename(columns={'OPEN':'TTF_OPEN'})
    TTF = TTF.dropna(subset=['TTF_OPEN'])
    TTF = TTF.rename(columns={'VOLUME':'TTF_VOLUME'})
    TTF = TTF.fillna(0)
    # display(TTF) 
    ########################MOMENTUM INDICATORS###############################################################
    TTF['TTF_ADX'] = talib.ADX(TTF['TTF_HIGH'],TTF['TTF_LOW'],TTF['TTF_CLOSE'], 14).astype(float)
    TTF['TTF_ADXR'] = talib.ADXR(TTF['TTF_HIGH'],TTF['TTF_LOW'],TTF['TTF_CLOSE'], 14).astype(float)
    TTF['TTF_APO'] = talib.APO(TTF['TTF_CLOSE'], 12,26,0).astype(float)
    TTF['TTF_AROON_DOWN'] = talib.AROON(TTF['TTF_HIGH'],TTF['TTF_LOW'],14)[0].astype(float)
    TTF['TTF_AROON_UP'] = talib.AROON(TTF['TTF_HIGH'],TTF['TTF_LOW'],14)[1].astype(float)
    TTF['TTF_AROONOSC'] = talib.AROONOSC(TTF['TTF_HIGH'],TTF['TTF_LOW'],14).astype(float)
    TTF['TTF_BOP'] = talib.BOP(TTF['TTF_OPEN'],TTF['TTF_HIGH'],TTF['TTF_LOW'],TTF['TTF_CLOSE']).astype(float)
    TTF['TTF_CCI'] = talib.CCI(TTF['TTF_HIGH'],TTF['TTF_LOW'],TTF['TTF_CLOSE'], 14).astype(float)
    TTF['TTF_CMO'] = talib.CMO(TTF['TTF_CLOSE'],14).astype(float)
    TTF['TTF_DX'] = talib.DX(TTF['TTF_HIGH'],TTF['TTF_LOW'],TTF['TTF_CLOSE'], 14).astype(float)
    TTF['TTF_MACD'] = talib.MACD(TTF['TTF_CLOSE'], 12, 26, 9)[0].astype(float)
    TTF['TTF_MACDSIGNALS'] = talib.MACD(TTF['TTF_CLOSE'], 12, 26, 9)[1].astype(float)
    TTF['TTF_MACDHIST'] = talib.MACD(TTF['TTF_CLOSE'], 12, 26, 9)[0].astype(float)
    TTF['TTF_MFI'] = talib.MFI(TTF['TTF_HIGH'],TTF['TTF_LOW'],TTF['TTF_CLOSE'],TTF['TTF_VOLUME'], 14).astype(float)
    TTF['TTF_MINUS_DI'] = talib.MINUS_DI(TTF['TTF_HIGH'],TTF['TTF_LOW'],TTF['TTF_CLOSE'], 14).astype(float)
    TTF['TTF_MINUS_DM'] = talib.MINUS_DM(TTF['TTF_HIGH'],TTF['TTF_LOW'],14).astype(float)
    TTF['TTF_MOM'] = talib.MOM(TTF['TTF_CLOSE'],14).astype(float)
    TTF['TTF_PLUS_DI'] = talib.PLUS_DI(TTF['TTF_HIGH'],TTF['TTF_LOW'],TTF['TTF_CLOSE'], 14).astype(float)
    TTF['TTF_PLUS_DM'] = talib.PLUS_DM(TTF['TTF_HIGH'],TTF['TTF_LOW'], 14).astype(float)
    TTF['TTF_PPO'] = talib.PPO(TTF['TTF_CLOSE'], 12, 26).astype(float)
    TTF['TTF_ROC'] = talib.ROC(TTF['TTF_CLOSE'], 10).astype(float)
    TTF['TTF_ROCP'] = talib.ROCP(TTF['TTF_CLOSE'], 10).astype(float)
    TTF['TTF_ROCR'] = talib.ROCR(TTF['TTF_CLOSE'], 10).astype(float)
    TTF['TTF_ROCR100'] = talib.ROCR100(TTF['TTF_CLOSE'], 10).astype(float)
    TTF['TTF_RSI'] = talib.RSI(TTF['TTF_CLOSE'],14).astype(float)
    TTF['TTF_SLOWK'] = talib.STOCH(TTF['TTF_HIGH'],TTF['TTF_LOW'],TTF['TTF_CLOSE'],5,3,0,3,0)[0].astype(float)
    TTF['TTF_SLOWD'] = talib.STOCH(TTF['TTF_HIGH'],TTF['TTF_LOW'],TTF['TTF_CLOSE'],5,3,0,3,0)[1].astype(float)
    TTF['TTF_FASTK'] = talib.STOCHF(TTF['TTF_HIGH'],TTF['TTF_LOW'],TTF['TTF_CLOSE'],5,3,0)[0].astype(float)
    TTF['TTF_FASTD'] = talib.STOCHF(TTF['TTF_HIGH'],TTF['TTF_LOW'],TTF['TTF_CLOSE'],5,3,0)[1].astype(float)
    TTF['TTF_STOCHRSI0'] = talib.STOCHRSI(TTF['TTF_CLOSE'],14,5,3,0)[0].astype(float)
    TTF['TTF_STOCHRSI1'] = talib.STOCHRSI(TTF['TTF_CLOSE'],14,5,3,0)[1].astype(float)
    TTF['TTF_TRIX'] = talib.TRIX(TTF['TTF_CLOSE'],30).astype(float)
    TTF['TTF_ULTOSC'] = talib.ULTOSC(TTF['TTF_HIGH'],TTF['TTF_LOW'],TTF['TTF_CLOSE'],7,14,28).astype(float)
    TTF['TTF_WILLR'] = talib.WILLR(TTF['TTF_HIGH'],TTF['TTF_LOW'],TTF['TTF_CLOSE'],14).astype(float)
    ########################MOMENTUM INDICATORS###############################################################

    ########################VOLUME INDICATORS###############################################################
    TTF['TTF_AD'] = talib.AD(TTF['TTF_HIGH'],TTF['TTF_LOW'],TTF['TTF_CLOSE'],TTF['TTF_VOLUME']).astype(float)
    TTF['TTF_ADOSC'] = talib.ADOSC(TTF['TTF_HIGH'],TTF['TTF_LOW'],TTF['TTF_CLOSE'],TTF['TTF_VOLUME'],3,10).astype(float)
    TTF['TTF_OBV'] = talib.OBV(TTF['TTF_CLOSE'],TTF['TTF_VOLUME']).astype(float)
    ########################VOLUME INDICATORS###############################################################

    ########################VOLATILITY INDICATORS###############################################################
    TTF['TTF_ATR'] = talib.ATR(TTF['TTF_HIGH'],TTF['TTF_LOW'],TTF['TTF_CLOSE'],14).astype(float)
    TTF['TTF_NATR'] = talib.NATR(TTF['TTF_HIGH'],TTF['TTF_LOW'],TTF['TTF_CLOSE'],14).astype(float)
    TTF['TTF_TRANGE'] = talib.TRANGE(TTF['TTF_HIGH'],TTF['TTF_LOW'],TTF['TTF_CLOSE']).astype(float)
    ########################VOLATILITY INDICATORS###############################################################

    ########################CYCLE INDICATORS###############################################################
    TTF['TTF_HT_DCPERIOD'] = talib.HT_DCPERIOD(TTF['TTF_CLOSE']).astype(float)
    TTF['TTF_HT_DCPHASE'] = talib.HT_DCPHASE(TTF['TTF_CLOSE']).astype(float)
    TTF['TTF_HT_PHASOR_0'] = talib.HT_PHASOR(TTF['TTF_CLOSE'])[0].astype(float)
    TTF['TTF_HT_PHASOR_1'] = talib.HT_PHASOR(TTF['TTF_CLOSE'])[1].astype(float)
    TTF['TTF_HT_SINE_0'] = talib.HT_SINE(TTF['TTF_CLOSE'])[0].astype(float)
    TTF['TTF_HT_SINE_1'] = talib.HT_SINE(TTF['TTF_CLOSE'])[1].astype(float)
    TTF['TTF_HT_TRENDMODE'] = talib.HT_TRENDMODE(TTF['TTF_CLOSE']).astype(float)
    ########################CYCLE INDICATORS###############################################################

    ########################PATTERN RECOGNITION###############################################################
    TTF['TTF_CDL2CROWS'] = talib.CDL2CROWS(TTF['TTF_OPEN'],TTF['TTF_HIGH'],TTF['TTF_LOW'],TTF['TTF_CLOSE']).astype(float)
    TTF['TTF_CDL3BLACKCROWS'] = talib.CDL3BLACKCROWS(TTF['TTF_OPEN'],TTF['TTF_HIGH'],TTF['TTF_LOW'],TTF['TTF_CLOSE']).astype(float)
    TTF['TTF_CDL3INSIDE'] = talib.CDL3INSIDE(TTF['TTF_OPEN'],TTF['TTF_HIGH'],TTF['TTF_LOW'],TTF['TTF_CLOSE']).astype(float)
    TTF['TTF_CDL3OUTSIDE'] = talib.CDL3OUTSIDE(TTF['TTF_OPEN'],TTF['TTF_HIGH'],TTF['TTF_LOW'],TTF['TTF_CLOSE']).astype(float)
    ########################PATTERN RECOGNITION###############################################################


    # display(TTF)

    brent=ek.get_timeseries('LCOc12',start_date=startdate,end_date=current_date) 
    brent = brent.rename(columns={'CLOSE':'brent_CLOSE'})
    brent = brent.dropna(subset=['brent_CLOSE'])
    brent['brent_RSI'] = talib.RSI(brent['brent_CLOSE'],14).astype(float)
    # display(brent)

    coal=ek.get_timeseries('TRAPI2Yc1',start_date=startdate,end_date=current_date)
    coal = coal.rename(columns={'CLOSE':'coal_CLOSE'})
    coal = coal.dropna(subset=['coal_CLOSE'])
    # display(coal)
    coal['coal_RSI'] = talib.RSI(coal['coal_CLOSE'],14).astype(float)
    # display(coal)

    CO2=ek.get_timeseries('CFI2c12',start_date=startdate,end_date=current_date)
    CO2 = CO2.rename(columns={'CLOSE':'CO2_CLOSE'})
    CO2 = CO2.dropna(subset=['CO2_CLOSE'])
    CO2['CO2_RSI'] = talib.RSI(CO2['CO2_CLOSE'],14).astype(float)
    # display(CO2)
    
    data = pd.merge(pd.DataFrame(TTF['TTF_CLOSE']), pd.DataFrame(TTF['TTF_RSI']), how = 'inner', left_index=True, right_index=True)
    data = pd.merge(data, pd.DataFrame(TTF['TTF_ADX']), how = 'inner', left_index=True, right_index=True)
    data = pd.merge(data, pd.DataFrame(TTF['TTF_ADXR']), how = 'inner', left_index=True, right_index=True)
    data = pd.merge(data, pd.DataFrame(TTF['TTF_CCI']), how = 'inner', left_index=True, right_index=True)
    data = pd.merge(data, pd.DataFrame(TTF['TTF_DX']), how = 'inner', left_index=True, right_index=True)
    data = pd.merge(data, pd.DataFrame(TTF['TTF_CMO']), how = 'inner', left_index=True, right_index=True)
    data = pd.merge(data, pd.DataFrame(TTF['TTF_APO']), how = 'inner', left_index=True, right_index=True)
    data = pd.merge(data, pd.DataFrame(TTF['TTF_AROON_DOWN']), how = 'inner', left_index=True, right_index=True)
    data = pd.merge(data, pd.DataFrame(TTF['TTF_AROON_UP']), how = 'inner', left_index=True, right_index=True)
    data = pd.merge(data, pd.DataFrame(TTF['TTF_AROONOSC']), how = 'inner', left_index=True, right_index=True)
    data = pd.merge(data, pd.DataFrame(TTF['TTF_BOP']), how = 'inner', left_index=True, right_index=True)
    data = pd.merge(data, pd.DataFrame(TTF['TTF_MACDSIGNALS']), how = 'inner', left_index=True, right_index=True)
    data = pd.merge(data, pd.DataFrame(TTF['TTF_MACDHIST']), how = 'inner', left_index=True, right_index=True)
    data = pd.merge(data, pd.DataFrame(TTF['TTF_MFI']), how = 'inner', left_index=True, right_index=True)
    data = pd.merge(data, pd.DataFrame(TTF['TTF_MINUS_DI']), how = 'inner', left_index=True, right_index=True)
    data = pd.merge(data, pd.DataFrame(TTF['TTF_MINUS_DM']), how = 'inner', left_index=True, right_index=True)
    data = pd.merge(data, pd.DataFrame(TTF['TTF_MOM']), how = 'inner', left_index=True, right_index=True)
    data = pd.merge(data, pd.DataFrame(TTF['TTF_PLUS_DI']), how = 'inner', left_index=True, right_index=True)
    data = pd.merge(data, pd.DataFrame(TTF['TTF_PLUS_DM']), how = 'inner', left_index=True, right_index=True)
    data = pd.merge(data, pd.DataFrame(TTF['TTF_PPO']), how = 'inner', left_index=True, right_index=True)
    data = pd.merge(data, pd.DataFrame(TTF['TTF_ROC']), how = 'inner', left_index=True, right_index=True)
    data = pd.merge(data, pd.DataFrame(TTF['TTF_ROCP']), how = 'inner', left_index=True, right_index=True)
    data = pd.merge(data, pd.DataFrame(TTF['TTF_ROCR']), how = 'inner', left_index=True, right_index=True)
    data = pd.merge(data, pd.DataFrame(TTF['TTF_ROCR100']), how = 'inner', left_index=True, right_index=True)
    data = pd.merge(data, pd.DataFrame(TTF['TTF_SLOWK']), how = 'inner', left_index=True, right_index=True)
    data = pd.merge(data, pd.DataFrame(TTF['TTF_SLOWD']), how = 'inner', left_index=True, right_index=True)
    data = pd.merge(data, pd.DataFrame(TTF['TTF_FASTK']), how = 'inner', left_index=True, right_index=True)
    data = pd.merge(data, pd.DataFrame(TTF['TTF_FASTD']), how = 'inner', left_index=True, right_index=True)
    data = pd.merge(data, pd.DataFrame(TTF['TTF_STOCHRSI0']), how = 'inner', left_index=True, right_index=True)
    data = pd.merge(data, pd.DataFrame(TTF['TTF_STOCHRSI1']), how = 'inner', left_index=True, right_index=True)
    data = pd.merge(data, pd.DataFrame(TTF['TTF_TRIX']), how = 'inner', left_index=True, right_index=True)
    data = pd.merge(data, pd.DataFrame(TTF['TTF_ULTOSC']), how = 'inner', left_index=True, right_index=True)
    data = pd.merge(data, pd.DataFrame(TTF['TTF_ULTOSC']), how = 'inner', left_index=True, right_index=True)
    data = pd.merge(data, pd.DataFrame(TTF['TTF_ULTOSC']), how = 'inner', left_index=True, right_index=True)
    data = pd.merge(data, pd.DataFrame(TTF['TTF_WILLR']), how = 'inner', left_index=True, right_index=True)
    data = pd.merge(data, pd.DataFrame(TTF['TTF_AD']), how = 'inner', left_index=True, right_index=True)
    data = pd.merge(data, pd.DataFrame(TTF['TTF_ADOSC']), how = 'inner', left_index=True, right_index=True)
    data = pd.merge(data, pd.DataFrame(TTF['TTF_OBV']), how = 'inner', left_index=True, right_index=True)
    data = pd.merge(data, pd.DataFrame(TTF['TTF_ATR']), how = 'inner', left_index=True, right_index=True)
    data = pd.merge(data, pd.DataFrame(TTF['TTF_NATR']), how = 'inner', left_index=True, right_index=True)
    data = pd.merge(data, pd.DataFrame(TTF['TTF_TRANGE']), how = 'inner', left_index=True, right_index=True)
    data = pd.merge(data, pd.DataFrame(TTF['TTF_HT_DCPERIOD']), how = 'inner', left_index=True, right_index=True)
    data = pd.merge(data, pd.DataFrame(TTF['TTF_HT_DCPHASE']), how = 'inner', left_index=True, right_index=True)
    data = pd.merge(data, pd.DataFrame(TTF['TTF_HT_PHASOR_0']), how = 'inner', left_index=True, right_index=True)
    data = pd.merge(data, pd.DataFrame(TTF['TTF_HT_PHASOR_1']), how = 'inner', left_index=True, right_index=True)
    data = pd.merge(data, pd.DataFrame(TTF['TTF_HT_SINE_0']), how = 'inner', left_index=True, right_index=True)
    data = pd.merge(data, pd.DataFrame(TTF['TTF_HT_SINE_1']), how = 'inner', left_index=True, right_index=True)
    data = pd.merge(data, pd.DataFrame(TTF['TTF_HT_TRENDMODE']), how = 'inner', left_index=True, right_index=True)
    data = pd.merge(data, pd.DataFrame(TTF['TTF_CDL3OUTSIDE']), how = 'inner', left_index=True, right_index=True)

    select = data.ix[:].dropna()
    data = data.dropna()

    data = data.dropna()
    return data