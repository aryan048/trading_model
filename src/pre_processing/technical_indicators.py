import sqlite3
import pandas as pd
import talib as ta
from sqlalchemy import create_engine
from tqdm import tqdm
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src import SMS_notifier


def create_technical_indicators(ticker, ticker_df, engine):
    indicators = {
        #overlap studies
        'upperband':lambda x: ta.BBANDS(x['close'], timeperiod=5, nbdevup=2, nbdevdn=0)[0],
        'middleband':lambda x: ta.BBANDS(x['close'], timeperiod=5, nbdevup=2, nbdevdn=0)[1],
        'lowerband':lambda x: ta.BBANDS(x['close'], timeperiod=5, nbdevup=2, nbdevdn=0)[2],
        'dema':lambda x: ta.DEMA(x['close'], timeperiod=5),
        'ema':lambda x: ta.EMA(x['close'], timeperiod=5),
        'ht_trendline':lambda x: ta.HT_TRENDLINE(x['close']),
        'kama':lambda x: ta.KAMA(x['close'], timeperiod=30),
        'ma':lambda x: ta.MA(x['close'], timeperiod=30, matype=0),
        'mama':lambda x: ta.MAMA(x['close'], fastlimit=0.5, slowlimit=0.05)[0],
        'fama':lambda x: ta.MAMA(x['close'], fastlimit=0.5, slowlimit=0.05)[1],
        #mavp = lambda x: ta.MAVP(x['close'], period=x.date, minperiod=2, maxperiod=30, matype=0),
        'midpoint':lambda x: ta.MIDPOINT(x['close'], timeperiod=14),
        'midprice':lambda x: ta.MIDPRICE(x['high'], x['low'], timeperiod=14),
        'sar':lambda x: ta.SAR(x['high'], x['low'], acceleration=0.02, maximum=0.2),
        'sarext':lambda x: ta.SAREXT(x['high'], x['low'], startvalue=0, offsetonreverse=0, accelerationinitlong=0.02, accelerationlong=0.02, accelerationmaxlong=0.2, accelerationinitshort=0.02, accelerationshort=0.02, accelerationmaxshort=0.2),
        'sma':lambda x: ta.SMA(x['close'], timeperiod=30),
        't3':lambda x: ta.T3(x['close'], timeperiod=5, vfactor=0),
        'tema':lambda x: ta.TEMA(x['close'], timeperiod=30),
        'trima':lambda x: ta.TRIMA(x['close'], timeperiod=30),
        'wma':lambda x: ta.WMA(x['close'], timeperiod=30),

        #momentum
        'adx':lambda x: ta.ADX(x['high'], x['low'], x['close'], timeperiod=14),
        'adxr':lambda x: ta.ADXR(x['high'], x['low'], x['close'], timeperiod=14),
        'apo':lambda x: ta.APO(x['close'], fastperiod=12, slowperiod=26, matype=0),
        'aroondown':lambda x: ta.AROON(x['high'], x['low'], timeperiod=14)[0],
        'aroonup':lambda x: ta.AROON(x['high'], x['low'], timeperiod=14)[1],
        'aroonosc':lambda x: ta.AROONOSC(x['high'], x['low'], timeperiod=14),
        'bop':lambda x: ta.BOP(x['open'], x['high'], x['low'], x['close']),
        'cci':lambda x: ta.CCI(x['high'], x['low'], x['close'], timeperiod=14),
        'cmo':lambda x: ta.CMO(x['close'], timeperiod=14),
        'dx':lambda x: ta.DX(x['high'], x['low'], x['close'], timeperiod=14),
        'macd':lambda x: ta.MACD(x['close'], fastperiod=12, slowperiod=26, signalperiod=9)[0],
        'macdsignal':lambda x: ta.MACD(x['close'], fastperiod=12, slowperiod=26, signalperiod=9)[1],
        'macdhist':lambda x: ta.MACD(x['close'], fastperiod=12, slowperiod=26, signalperiod=9)[2],
        'macdext':lambda x: ta.MACDEXT(x['close'], fastperiod=12, fastmatype=0, slowperiod=26, slowmatype=0, signalperiod=9, signalmatype=0)[0],
        'macdfix':lambda x: ta.MACDFIX(x['close'], signalperiod=9)[0],
        'mfi':lambda x: ta.MFI(x['high'], x['low'], x['close'], x['volume'], timeperiod=14),
        'minus_di':lambda x: ta.MINUS_DI(x['high'], x['low'], x['close'], timeperiod=14),
        'minus_dm':lambda x: ta.MINUS_DM(x['high'], x['low'], timeperiod=14),
        'mom':lambda x: ta.MOM(x['close'], timeperiod=10),
        'plus_di':lambda x: ta.PLUS_DI(x['high'], x['low'], x['close'], timeperiod=14),
        'plus_dm':lambda x: ta.PLUS_DM(x['high'], x['low'], timeperiod=14),
        'ppo':lambda x: ta.PPO(x['close'], fastperiod=12, slowperiod=26, matype=0),
        'roc':lambda x: ta.ROC(x['close'], timeperiod=10),
        'rocp':lambda x: ta.ROCP(x['close'], timeperiod=10),
        'rocr':lambda x: ta.ROCR(x['close'], timeperiod=10),
        'rocr100':lambda x: ta.ROCR100(x['close'], timeperiod=10),
        'rsi':lambda x: ta.RSI(x['close'], timeperiod=14),
        'slowk':lambda x: ta.STOCH(x['high'], x['low'], x['close'], fastk_period=5, slowk_period=3, slowk_matype=0, slowd_period=3, slowd_matype=0)[0],
        'slowd':lambda x: ta.STOCH(x['high'], x['low'], x['close'], fastk_period=5, slowk_period=3, slowk_matype=0, slowd_period=3, slowd_matype=0)[1],
        'fastk':lambda x: ta.STOCHF(x['high'], x['low'], x['close'], fastk_period=5, fastd_period=3, fastd_matype=0)[0],
        'fastd':lambda x: ta.STOCHF(x['high'], x['low'], x['close'], fastk_period=5, fastd_period=3, fastd_matype=0)[1],
        'sto_rsi_k':lambda x: ta.STOCHRSI(x['close'], timeperiod=14, fastk_period=5, fastd_period=3, fastd_matype=0)[0],
        'sto_rsi_d':lambda x: ta.STOCHRSI(x['close'], timeperiod=14, fastk_period=5, fastd_period=3, fastd_matype=0)[1],
        'trix':lambda x: ta.TRIX(x['close'], timeperiod=30),
        'ultosc':lambda x: ta.ULTOSC(x['high'], x['low'], x['close'], timeperiod1=7, timeperiod2=14, timeperiod3=28),
        'willr':lambda x: ta.WILLR(x['high'], x['low'], x['close'], timeperiod=14),

        #volume
        'ad':lambda x: ta.AD(x['high'], x['low'], x['close'], x['volume']),
        'adosc':lambda x: ta.ADOSC(x['high'], x['low'], x['close'], x['volume'], fastperiod=3, slowperiod=10),
        'obv':lambda x: ta.OBV(x['close'], x['volume']),

        #volatility
        'atr':lambda x: ta.ATR(x['high'], x['low'], x['close'], timeperiod=14),
        'natr':lambda x: ta.NATR(x['high'], x['low'], x['close'], timeperiod=14),
        'trange':lambda x: ta.TRANGE(x['high'], x['low'], x['close']),

        #price transforms
        'avgprice':lambda x: ta.AVGPRICE(x['open'], x['high'], x['low'], x['close']),
        'medprice':lambda x: ta.MEDPRICE(x['high'], x['low']),
        'typprice':lambda x: ta.TYPPRICE(x['high'], x['low'], x['close']),
        'wclprice':lambda x: ta.WCLPRICE(x['high'], x['low'], x['close']),

        #cycle indicators
        'ht_dcperiod':lambda x: ta.HT_DCPERIOD(x['close']),
        'ht_dcphase':lambda x: ta.HT_DCPHASE(x['close']),
        'ht_inphase':lambda x: ta.HT_PHASOR(x['close'])[0],
        'ht_quadrature':lambda x: ta.HT_PHASOR(x['close'])[1],
        'ht_sine':lambda x: ta.HT_SINE(x['close'])[0],
        'ht_leadsine':lambda x: ta.HT_SINE(x['close'])[1],
        'ht_trendmode':lambda x: ta.HT_TRENDMODE(x['close']),

        #pattern recognition
        'cdl2crows':lambda x: ta.CDL2CROWS(x['open'], x['high'], x['low'], x['close']),
        'cdl3blackcrows':lambda x: ta.CDL3BLACKCROWS(x['open'], x['high'], x['low'], x['close']),
        'cdl3inside':lambda x: ta.CDL3INSIDE(x['open'], x['high'], x['low'], x['close']),
        'cdl3linestrike':lambda x: ta.CDL3LINESTRIKE(x['open'], x['high'], x['low'], x['close']),
        'cdl3outside':lambda x: ta.CDL3OUTSIDE(x['open'], x['high'], x['low'], x['close']),
        'cdl3starsinsouth':lambda x: ta.CDL3STARSINSOUTH(x['open'], x['high'], x['low'], x['close']),
        'cdl3whitesoldiers':lambda x: ta.CDL3WHITESOLDIERS(x['open'], x['high'], x['low'], x['close']),
        'cdlabandonedbaby':lambda x: ta.CDLABANDONEDBABY(x['open'], x['high'], x['low'], x['close'], penetration=0),
        'cdladvanceblock':lambda x: ta.CDLADVANCEBLOCK(x['open'], x['high'], x['low'], x['close']),
        #cdlbelthold:lambda x: ta.CDLBELTHOLD(x['open'], x['high'], x['low'], x['close']),
        'cdlbreakaway':lambda x: ta.CDLBREAKAWAY(x['open'], x['high'], x['low'], x['close']),
        'cdlclosingmarubozu':lambda x: ta.CDLCLOSINGMARUBOZU(x['open'], x['high'], x['low'], x['close']),
        'cdlconcealbabyswall':lambda x: ta.CDLCONCEALBABYSWALL(x['open'], x['high'], x['low'], x['close']),
        'cdlcounterattack':lambda x: ta.CDLCOUNTERATTACK(x['open'], x['high'], x['low'], x['close']),
        'cdldarkcloudcover':lambda x: ta.CDLDARKCLOUDCOVER(x['open'], x['high'], x['low'], x['close'], penetration=0),
        'cdldoji':lambda x: ta.CDLDOJI(x['open'], x['high'], x['low'], x['close']),
        'cdldojistar':lambda x: ta.CDLDOJISTAR(x['open'], x['high'], x['low'], x['close']),
        'cdldragonflydoji':lambda x: ta.CDLDRAGONFLYDOJI(x['open'], x['high'], x['low'], x['close']),
        'cdlengulfing':lambda x: ta.CDLENGULFING(x['open'], x['high'], x['low'], x['close']),
        'cdleveningdojistar':lambda x: ta.CDLEVENINGDOJISTAR(x['open'], x['high'], x['low'], x['close'], penetration=0),
        'cdleveningstar':lambda x: ta.CDLEVENINGSTAR(x['open'], x['high'], x['low'], x['close'], penetration=0),
        'cdlgapsidesidewhite':lambda x: ta.CDLGAPSIDESIDEWHITE(x['open'], x['high'], x['low'], x['close']),
        'cdlgravestonedoji':lambda x: ta.CDLGRAVESTONEDOJI(x['open'], x['high'], x['low'], x['close']),
        'cdlhammer':lambda x: ta.CDLHAMMER(x['open'], x['high'], x['low'], x['close']),
        'cdlhangingman':lambda x: ta.CDLHANGINGMAN(x['open'], x['high'], x['low'], x['close']),
        'cdlharami':lambda x: ta.CDLHARAMI(x['open'], x['high'], x['low'], x['close']),
        'cdlharamicross':lambda x: ta.CDLHARAMICROSS(x['open'], x['high'], x['low'], x['close']),
        'cdlhighwave':lambda x: ta.CDLHIGHWAVE(x['open'], x['high'], x['low'], x['close']),
        'cdlhikkake':lambda x: ta.CDLHIKKAKE(x['open'], x['high'], x['low'], x['close']),
        'cdlhikkakemod':lambda x: ta.CDLHIKKAKEMOD(x['open'], x['high'], x['low'], x['close']),
        'cdlhomingpigeon':lambda x: ta.CDLHOMINGPIGEON(x['open'], x['high'], x['low'], x['close']),
        'cdlidentical3crows':lambda x: ta.CDLIDENTICAL3CROWS(x['open'], x['high'], x['low'], x['close']),
        'cdlinneck':lambda x: ta.CDLINNECK(x['open'], x['high'], x['low'], x['close']),
        'cdlinvertedhammer':lambda x: ta.CDLINVERTEDHAMMER(x['open'], x['high'], x['low'], x['close']),
        'cdlkicking':lambda x: ta.CDLKICKING(x['open'], x['high'], x['low'], x['close']),
        'cdlkickingbylength':lambda x: ta.CDLKICKINGBYLENGTH(x['open'], x['high'], x['low'], x['close']),
        'cdlladderbottom':lambda x: ta.CDLLADDERBOTTOM(x['open'], x['high'], x['low'], x['close']),
        'cdllongleggeddoji':lambda x: ta.CDLLONGLEGGEDDOJI(x['open'], x['high'], x['low'], x['close']),
        'cdllongline':lambda x: ta.CDLLONGLINE(x['open'], x['high'], x['low'], x['close']),
        'cdlmarubozu':lambda x: ta.CDLMARUBOZU(x['open'], x['high'], x['low'], x['close']),
        'cdlmatchinglow':lambda x: ta.CDLMATCHINGLOW(x['open'], x['high'], x['low'], x['close']),
        'cdlmathold':lambda x: ta.CDLMATHOLD(x['open'], x['high'], x['low'], x['close'], penetration=0),
        'cdlmorningdojistar':lambda x: ta.CDLMORNINGDOJISTAR(x['open'], x['high'], x['low'], x['close'], penetration=0),
        'cdlmorningstar':lambda x: ta.CDLMORNINGSTAR(x['open'], x['high'], x['low'], x['close'], penetration=0),
        'cdlonneck':lambda x: ta.CDLONNECK(x['open'], x['high'], x['low'], x['close']),
        'cdlpiercing':lambda x: ta.CDLPIERCING(x['open'], x['high'], x['low'], x['close']),
        'cdlrickshawman':lambda x: ta.CDLRICKSHAWMAN(x['open'], x['high'], x['low'], x['close']),
        'cdlrisefall3methods':lambda x: ta.CDLRISEFALL3METHODS(x['open'], x['high'], x['low'], x['close']),
        'cdlseparatinglines':lambda x: ta.CDLSEPARATINGLINES(x['open'], x['high'], x['low'], x['close']),
        'cdlshootingstar':lambda x: ta.CDLSHOOTINGSTAR(x['open'], x['high'], x['low'], x['close']),
        'cdlshortline':lambda x: ta.CDLSHORTLINE(x['open'], x['high'], x['low'], x['close']),
        'cdlspinningtop':lambda x: ta.CDLSPINNINGTOP(x['open'], x['high'], x['low'], x['close']),
        'cdlstallepattern':lambda x: ta.CDLSTALLEDPATTERN(x['open'], x['high'], x['low'], x['close']),
        'cdlsticksandwich':lambda x: ta.CDLSTICKSANDWICH(x['open'], x['high'], x['low'], x['close']),
        'cdltakuri':lambda x: ta.CDLTAKURI(x['open'], x['high'], x['low'], x['close']),
        'cdltasukigap':lambda x: ta.CDLTASUKIGAP(x['open'], x['high'], x['low'], x['close']),
        'cdlthrusting':lambda x: ta.CDLTHRUSTING(x['open'], x['high'], x['low'], x['close']),
        'cdltristar':lambda x: ta.CDLTRISTAR(x['open'], x['high'], x['low'], x['close']),
        'cdlunique3river':lambda x: ta.CDLUNIQUE3RIVER(x['open'], x['high'], x['low'], x['close']),
        'cdlupsidegap2crows':lambda x: ta.CDLUPSIDEGAP2CROWS(x['open'], x['high'], x['low'], x['close']),
        'cdlxsidegap3methods':lambda x: ta.CDLXSIDEGAP3METHODS(x['open'], x['high'], x['low'], x['close']),

        #statistics
        #add these later with market close price
        # beta:lambda x: ta.BETA(x['real0'], x['real1'], timeperiod=5),
        # correl:lambda x: ta.CORREL(x['real0'], x['real1'], timeperiod=30),
        'linearreg':lambda x: ta.LINEARREG(x['close'], timeperiod=14),
        'linearreg_angle':lambda x: ta.LINEARREG_ANGLE(x['close'], timeperiod=14),
        'linearreg_intercept':lambda x: ta.LINEARREG_INTERCEPT(x['close'], timeperiod=14),
        'linearreg_slope':lambda x: ta.LINEARREG_SLOPE(x['close'], timeperiod=14),
        'stddev':lambda x: ta.STDDEV(x['close'], timeperiod=5, nbdev=1),
        'tsf':lambda x: ta.TSF(x['close'], timeperiod=14),
        'var':lambda x: ta.VAR(x['close'], timeperiod=5, nbdev=1),

        #math transformations
        'acos':lambda x: ta.ACOS(x['close']),
        'asin':lambda x: ta.ASIN(x['close']),
        'atan':lambda x: ta.ATAN(x['close']),
        'ceil':lambda x: ta.CEIL(x['close']),
        'cos':lambda x: ta.COS(x['close']),
        'cosh':lambda x: ta.COSH(x['close']),
        'exp':lambda x: ta.EXP(x['close']),
        'floor':lambda x: ta.FLOOR(x['close']),
        'ln':lambda x: ta.LN(x['close']),
        'log10':lambda x: ta.LOG10(x['close']),
        'sin':lambda x: ta.SIN(x['close']),
        'sinh':lambda x: ta.SINH(x['close']),
        'sqrt':lambda x: ta.SQRT(x['close']),
        'tan':lambda x: ta.TAN(x['close']),
        'tanh':lambda x: ta.TANH(x['close']),

        #math operators
        'add':lambda x: ta.ADD(x['close'], x['close']),
        'div':lambda x: ta.DIV(x['close'], x['close']),
        'max_val':lambda x: ta.MAX(x['close'], timeperiod=30),
        'maxindex':lambda x: ta.MAXINDEX(x['close'], timeperiod=30),
        'min_val':lambda x: ta.MIN(x['close'], timeperiod=30),
        'minindex':lambda x: ta.MININDEX(x['close'], timeperiod=30),
        #minmax = lambda x: ta.MINMAX(x['close'], timeperiod=30),
        #minmaxindex = lambda x= ta.MINMAXINDEX(x['close'], timeperiod=30),
        'mult':lambda x: ta.MULT(x['close'], x['close']),
        'sub':lambda x: ta.SUB(x['close'], x['close']),
        'sum_val':lambda x: ta.SUM(x['close'], timeperiod=30)
    }
    new_columns = {key: func(ticker_df) for key, func in indicators.items()}
    ticker_df = pd.concat([ticker_df, pd.DataFrame(new_columns)], axis=1)

    ticker_df.to_sql(ticker, engine, if_exists='replace', index=False)
        
if __name__ == "__main__":
    database = 'src/pre_processing/stock_data/stock_data.db'
    engine = create_engine(f'sqlite:///{database}')
    create_technical_indicators(engine, ['AAPL'])

