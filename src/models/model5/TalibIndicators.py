import talib as ta
class TalibIndicators():
    """Sticking with indicators I actually know for now, can add more and scale later"""
    def __init__(self, df):
        self.df = df
        self.features = []
        self.overlap_studies()
        self.momentum_indicators()
        self.volume_indicators()
        self.cycle_indicators()
        self.price_transformations()
        self.volatility_indicators()
        self.pattern_recognition()
        self.statistic_functions()
        self.df = self.df.dropna()

    def overlap_studies(self):
        self.df['ema'] = ta.EMA(self.df['adjusted_close'], timeperiod=30)
        self.df['ma'] = ta.MA(self.df['adjusted_close'], timeperiod=30)

        self.features += ['ema', 'ma']

    def momentum_indicators(self):
        self.df['macd'], self.df['macdsignal'], self.df['macdhist'] = ta.MACD(self.df['adjusted_close'], fastperiod=12, slowperiod=26, signalperiod=9)
        self.df['rsi'] = ta.RSI(self.df['adjusted_close'], timeperiod=14)

        self.features += ['macd', 'rsi']

    def volume_indicators(self):
        pass

    def cycle_indicators(self):
        pass

    def price_transformations(self):
        self.df['avgprice'] = ta.AVGPRICE(self.df['open'], self.df['high'], self.df['low'], self.df['adjusted_close'])

        self.features += ['avgprice']

    def volatility_indicators(self):
        pass

    def pattern_recognition(self):
        self.df['cdl2crows'] = ta.CDL2CROWS(self.df['open'], self.df['high'], self.df['low'], self.df['adjusted_close'])
        self.df['cdl3blackcrows'] = ta.CDL3BLACKCROWS(self.df['open'], self.df['high'], self.df['low'], self.df['adjusted_close'])
        self.df['cdl3inside'] = ta.CDL3INSIDE(self.df['open'], self.df['high'], self.df['low'], self.df['adjusted_close'])
        self.df['cdl3linestrike'] = ta.CDL3LINESTRIKE(self.df['open'], self.df['high'], self.df['low'], self.df['adjusted_close'])
        self.df['cdl3outside'] = ta.CDL3OUTSIDE(self.df['open'], self.df['high'], self.df['low'], self.df['adjusted_close'])
        self.df['cdl3starsinsouth'] = ta.CDL3STARSINSOUTH(self.df['open'], self.df['high'], self.df['low'], self.df['adjusted_close'])
        self.df['cdl3whitesoldiers'] = ta.CDL3WHITESOLDIERS(self.df['open'], self.df['high'], self.df['low'], self.df['adjusted_close'])
        self.df['cdlabandonedbaby'] = ta.CDLABANDONEDBABY(self.df['open'], self.df['high'], self.df['low'], self.df['adjusted_close'])
        self.df['cdladvanceblock'] = ta.CDLADVANCEBLOCK(self.df['open'], self.df['high'], self.df['low'], self.df['adjusted_close'])
        self.df['cdlbelthold'] = ta.CDLBELTHOLD(self.df['open'], self.df['high'], self.df['low'], self.df['adjusted_close'])
        self.df['cdlbreakaway'] = ta.CDLBREAKAWAY(self.df['open'], self.df['high'], self.df['low'], self.df['adjusted_close'])
        self.df['cdlclosingmarubozu'] = ta.CDLCLOSINGMARUBOZU(self.df['open'], self.df['high'], self.df['low'], self.df['adjusted_close'])
        self.df['cdlconcealbabyswall'] = ta.CDLCONCEALBABYSWALL(self.df['open'], self.df['high'], self.df['low'], self.df['adjusted_close'])
        self.df['cdlcounterattack'] = ta.CDLCOUNTERATTACK(self.df['open'], self.df['high'], self.df['low'], self.df['adjusted_close'])
        self.df['cdldarkcloudcover'] = ta.CDLDARKCLOUDCOVER(self.df['open'], self.df['high'], self.df['low'], self.df['adjusted_close'])
        self.df['cdldoji'] = ta.CDLDOJI(self.df['open'], self.df['high'], self.df['low'], self.df['adjusted_close'])
        self.df['cdldojistar'] = ta.CDLDOJISTAR(self.df['open'], self.df['high'], self.df['low'], self.df['adjusted_close'])
        self.df['cdlengulfing'] = ta.CDLENGULFING(self.df['open'], self.df['high'], self.df['low'], self.df['adjusted_close'])
        self.df['cdleveningdojistar'] = ta.CDLEVENINGDOJISTAR(self.df['open'], self.df['high'], self.df['low'], self.df['adjusted_close'])
        self.df['cdleveningstar'] = ta.CDLEVENINGSTAR(self.df['open'], self.df['high'], self.df['low'], self.df['adjusted_close'])
        self.df['cdlgapsidesidewhite'] = ta.CDLGAPSIDESIDEWHITE(self.df['open'], self.df['high'], self.df['low'], self.df['adjusted_close'])
        self.df['cdlgravestonedoji'] = ta.CDLGRAVESTONEDOJI(self.df['open'], self.df['high'], self.df['low'], self.df['adjusted_close'])
        self.df['cdlhammer'] = ta.CDLHAMMER(self.df['open'], self.df['high'], self.df['low'], self.df['adjusted_close'])
        self.df['cdlhangingman'] = ta.CDLHANGINGMAN(self.df['open'], self.df['high'], self.df['low'], self.df['adjusted_close'])
        self.df['cdlharami'] = ta.CDLHARAMI(self.df['open'], self.df['high'], self.df['low'], self.df['adjusted_close'])
        self.df['cdlharamicross'] = ta.CDLHARAMICROSS(self.df['open'], self.df['high'], self.df['low'], self.df['adjusted_close'])
        self.df['cdlhighwave'] = ta.CDLHIGHWAVE(self.df['open'], self.df['high'], self.df['low'], self.df['adjusted_close'])
        self.df['cdlhikkake'] = ta.CDLHIKKAKE(self.df['open'], self.df['high'], self.df['low'], self.df['adjusted_close'])
        self.df['cdlhikkakemod'] = ta.CDLHIKKAKEMOD(self.df['open'], self.df['high'], self.df['low'], self.df['adjusted_close'])
        self.df['cdlhomingpigeon'] = ta.CDLHOMINGPIGEON(self.df['open'], self.df['high'], self.df['low'], self.df['adjusted_close'])
        self.df['cdlidentical3crows'] = ta.CDLIDENTICAL3CROWS(self.df['open'], self.df['high'], self.df['low'], self.df['adjusted_close'])
        self.df['cdlinneck'] = ta.CDLINNECK(self.df['open'], self.df['high'], self.df['low'], self.df['adjusted_close'])
        self.df['cdlinvertedhammer'] = ta.CDLINVERTEDHAMMER(self.df['open'], self.df['high'], self.df['low'], self.df['adjusted_close'])
        self.df['cdlkicking'] = ta.CDLKICKING(self.df['open'], self.df['high'], self.df['low'], self.df['adjusted_close'])
        self.df['cdlkickingbylength'] = ta.CDLKICKINGBYLENGTH(self.df['open'], self.df['high'], self.df['low'], self.df['adjusted_close'])
        self.df['cdlladderbottom'] = ta.CDLLADDERBOTTOM(self.df['open'], self.df['high'], self.df['low'], self.df['adjusted_close'])
        self.df['cdllongleggeddoji'] = ta.CDLLONGLEGGEDDOJI(self.df['open'], self.df['high'], self.df['low'], self.df['adjusted_close'])
        self.df['cdllongline'] = ta.CDLLONGLINE(self.df['open'], self.df['high'], self.df['low'], self.df['adjusted_close'])
        self.df['cdlmarubozu'] = ta.CDLMARUBOZU(self.df['open'], self.df['high'], self.df['low'], self.df['adjusted_close'])
        self.df['cdlmatchinglow'] = ta.CDLMATCHINGLOW(self.df['open'], self.df['high'], self.df['low'], self.df['adjusted_close'])
        self.df['cdlmathold'] = ta.CDLMATHOLD(self.df['open'], self.df['high'], self.df['low'], self.df['adjusted_close'])
        self.df['cdlmorningdojistar'] = ta.CDLMORNINGDOJISTAR(self.df['open'], self.df['high'], self.df['low'], self.df['adjusted_close'])
        self.df['cdlmorningstar'] = ta.CDLMORNINGSTAR(self.df['open'], self.df['high'], self.df['low'], self.df['adjusted_close'])
        self.df['cdlonneck'] = ta.CDLONNECK(self.df['open'], self.df['high'], self.df['low'], self.df['adjusted_close'])
        self.df['cdlpiercing'] = ta.CDLPIERCING(self.df['open'], self.df['high'], self.df['low'], self.df['adjusted_close'])
        self.df['cdlrickshawman'] = ta.CDLRICKSHAWMAN(self.df['open'], self.df['high'], self.df['low'], self.df['adjusted_close'])
        self.df['cdlrisefallang3methods'] = ta.CDLRISEFALL3METHODS(self.df['open'], self.df['high'], self.df['low'], self.df['adjusted_close'])
        self.df['cdlseparatinglines'] = ta.CDLSEPARATINGLINES(self.df['open'], self.df['high'], self.df['low'], self.df['adjusted_close'])
        self.df['cdlshootingstar'] = ta.CDLSHOOTINGSTAR(self.df['open'], self.df['high'], self.df['low'], self.df['adjusted_close'])
        self.df['cdlshortline'] = ta.CDLSHORTLINE(self.df['open'], self.df['high'], self.df['low'], self.df['adjusted_close'])
        self.df['cdlspinningtop'] = ta.CDLSPINNINGTOP(self.df['open'], self.df['high'], self.df['low'], self.df['adjusted_close'])
        self.df['cdlstalledpattern'] = ta.CDLSTALLEDPATTERN(self.df['open'], self.df['high'], self.df['low'], self.df['adjusted_close'])
        self.df['cdlsticksandwich'] = ta.CDLSTICKSANDWICH(self.df['open'], self.df['high'], self.df['low'], self.df['adjusted_close'])
        self.df['cdltakuri'] = ta.CDLTAKURI(self.df['open'], self.df['high'], self.df['low'], self.df['adjusted_close'])
        self.df['cdltasukigap'] = ta.CDLTASUKIGAP(self.df['open'], self.df['high'], self.df['low'], self.df['adjusted_close'])
        self.df['cdlthrusting'] = ta.CDLTHRUSTING(self.df['open'], self.df['high'], self.df['low'], self.df['adjusted_close'])
        self.df['cdltristar'] = ta.CDLTRISTAR(self.df['open'], self.df['high'], self.df['low'], self.df['adjusted_close'])
        self.df['cdlunique3river'] = ta.CDLUNIQUE3RIVER(self.df['open'], self.df['high'], self.df['low'], self.df['adjusted_close'])
        self.df['cdlupsidegap2crows'] = ta.CDLUPSIDEGAP2CROWS(self.df['open'], self.df['high'], self.df['low'], self.df['adjusted_close'])
        self.df['cdlxsidegap3methods'] = ta.CDLXSIDEGAP3METHODS(self.df['open'], self.df['high'], self.df['low'], self.df['adjusted_close'])

        self.features += ['cdl2crows', 'cdl3blackcrows', 'cdl3inside', 'cdl3linestrike', 'cdl3outside', 'cdl3starsinsouth', 
                              'cdl3whitesoldiers', 'cdlabandonedbaby', 'cdladvanceblock', 'cdlbelthold', 'cdlbreakaway', 'cdlclosingmarubozu', 
                              'cdlconcealbabyswall', 'cdlcounterattack', 'cdldarkcloudcover', 'cdldoji', 'cdldojistar', 'cdlengulfing', 
                              'cdleveningdojistar', 'cdleveningstar', 'cdlgapsidesidewhite', 'cdlgravestonedoji', 'cdlhammer', 
                              'cdlhangingman', 'cdlharami', 'cdlharamicross', 'cdlhighwave', 'cdlhikkake', 'cdlhikkakemod', 'cdlhomingpigeon', 
                              'cdlidentical3crows', 'cdlinneck', 'cdlinvertedhammer', 'cdlkicking', 'cdlkickingbylength', 'cdlladderbottom', 
                              'cdllongleggeddoji', 'cdllongline', 'cdlmarubozu', 'cdlmatchinglow', 'cdlmathold', 'cdlmorningdojistar',
                              'cdlmorningstar', 'cdlonneck', 'cdlpiercing', 'cdlrickshawman', 'cdlrisefallang3methods', 'cdlseparatinglines',
                              'cdlshootingstar', 'cdlshortline', 'cdlspinningtop', 'cdlstalledpattern', 'cdlsticksandwich', 'cdltakuri',
                              'cdltasukigap', 'cdlthrusting', 'cdltristar', 'cdlunique3river', 'cdlupsidegap2crows', 'cdlxsidegap3methods']

    def statistic_functions(self):
        self.df['linearreg'] = ta.LINEARREG(self.df['adjusted_close'], timeperiod=14)

        self.features += ['linearreg']