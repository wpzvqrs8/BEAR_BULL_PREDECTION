import pandas as pd
f = r'C:\Users\Admin\PyCharmMiscProject\pythonprojects\my_pro\AI_ML_DL\STOCKMARKET_PREDICTOR_DEMO\datas\BTCUSDT-1m-2017-08.csv'
df = pd.read_csv(f, header=None, nrows=3)
print(df.to_string())
print("Shape:", df.shape)
