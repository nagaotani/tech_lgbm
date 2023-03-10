
import streamlit as st

import pandas_datareader.data as web
import numpy as np
import pandas as pd
import datetime

from sklearn.preprocessing import StandardScaler

import mplfinance as mpf

# 警告の無効化
import warnings
warnings.simplefilter('ignore')
#warnings.filterwarnings('ignore')

from sklearn.cluster import KMeans
from sklearn import metrics

import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist

import matplotlib.pyplot as plt # グラフ描画用
import seaborn as sns; sns.set() # グラフ描画用
import warnings # 実行に関係ない警告を無視
warnings.filterwarnings('ignore')
import lightgbm as lgb #LightGBM
from sklearn import datasets
from sklearn.model_selection import train_test_split # データセット分割用
from sklearn.metrics import mean_squared_error # モデル評価用(平均二乗誤差)
from sklearn.metrics import r2_score # モデル評価用(決定係数)


st.title('株式トレードサポートアプリ')
st.header('過去の株価データから特徴量を生成しクラスタリング、\
前日の株価の状況が属するクラスター内のデータをもとにLightGBMで回帰を行う')

#st.sidebar.header('銘柄コード入力')
#code = st.sidebar.text_input('銘柄コード（4桁、半角）') + '.T'

code = '9107'

code = st.text_input('銘柄コード入力（半角数字４ケタ）：例 ソニー 6758　トヨタ 7203　\
            日経平均 ^N225　S&P500 ^GSPC など', '9107')

if code[0] != '^':
    code = code + '.JP'

if len(code) < 3:
    st.warning('入力お願いします')
    # 条件を満たないときは処理を停止する
    st.stop()

# １、データ取得

#start = datetime.date(2017,1,1)
start = datetime.date(2015,1,1)
end = datetime.date.today()

#code = '^NKX'

df_origin = web.DataReader(code, 'stooq', start, end)
#df = web.DataReader('7203.JP', 'stooq', start, end)
#df = web.DataReader('7203.T', 'yahoo',start, end)


df_origin.sort_index(inplace=True)


fig1 = mpf.figure(figsize=(10, 10),style='yahoo')
ax1 = fig1.add_subplot(1,1,1)
mpf.plot(df_origin.iloc[-250:,:], ax=ax1, style='yahoo', type='candle', xrotation=30)

st.pyplot(fig1)

# ２，テクニカル計算

def technical_calc(df):
    df = df.sort_values('Date', ascending=True)
    
    #リターンの計算
    
    df['1day_return'] = df['Close'].pct_change(1) #5
#    df['3day_return'] = df['Close'].pct_change(3) #6
    df['5day_return'] = df['Close'].pct_change(5) #7
    df['10day_return'] = df['Close'].pct_change(10) #8
    df['25day_return'] = df['Close'].pct_change(25) #9

    #移動平均からの乖離計算
    df['5ma'] = df['Close'].rolling(5).mean() / df['Close'] - 1 #10
    df['10ma'] = df['Close'].rolling(10).mean() / df['Close'] - 1 #10
    df['20ma'] = df['Close'].rolling(20).mean() / df['Close'] - 1 #10
#    df['40ma'] = df['Close'].rolling(40).mean() / df['Close'] - 1 #11
    df['60ma'] = df['Close'].rolling(60).mean() / df['Close'] - 1 #12
#    df['120ma'] = df['Close'].rolling(120).mean() / df['Close'] - 1 #13
#    df['240ma'] = df['Close'].rolling(240).mean() / df['Close'] - 1 #14

    #過去N日間の高値、安値からの乖離
    df['5high'] = df['High'].rolling(5).max() / df['Close'] - 1 #15
    df['5low'] = df['Low'].rolling(5).min() / df['Close'] - 1 #16
    df['10high'] = df['High'].rolling(10).max() / df['Close'] - 1 #17
    df['10low'] = df['Low'].rolling(10).min() / df['Close'] - 1 #18
    df['20high'] = df['High'].rolling(20).max() / df['Close'] - 1 #19
    df['20low'] = df['Low'].rolling(20).min() / df['Close'] - 1 #20
#    df['30high'] = df['High'].rolling(30).max() / df['Close'] - 1 #21
#    df['30low'] = df['Low'].rolling(30).min() / df['Close'] - 1 #22
    df['60high'] = df['High'].rolling(60).max() / df['Close'] - 1 #23
    df['60low'] = df['Low'].rolling(60).min() / df['Close'] - 1 #24
    
    #ATRの計算
    df['TR'] = np.maximum(df['High'].values - df['Close'].shift(1).values, df['Close'].shift(1).values 
                           - df['Low'].values, df['High'].values - df['Low'].values) / df['Close'] #25
    #true_range_calc(df['High'].values, df['Low'].values, df['Close'].shift(1).values) #25
    df['ATR5'] = df['TR'].rolling(5).mean() #26
    df['ATR10'] = df['TR'].rolling(10).mean() #27
    df['ATR20'] = df['TR'].rolling(20).mean() #28
#    df['ATR40'] = df['TR'].rolling(40).mean() #29
    df['ATR60'] = df['TR'].rolling(60).mean() #30

#  予測ターゲットの計算
    df['1day_result'] = df['Close'].pct_change(-1) #31
    df['3day_result'] = df['Close'].pct_change(-3) #32
    df['5day_result'] = df['Close'].pct_change(-5) #33
    df['10day_result'] = df['Close'].pct_change(-10) #34
#    df['25day_result'] = df['Close'].pct_change(-25) #35

    return df

df = technical_calc(df_origin.copy())

# クラスタリング

# データの特徴量とターゲットに切り分け

df_x = df.dropna().iloc[:, 5:-4]
df_t = df.dropna().iloc[:, -4:]

# データの標準化



# 特徴量の抽出
X = df_x.values

# StandardScalerによる標準化
sc = StandardScaler()
X_std = sc.fit_transform(X)

df_x_std = pd.DataFrame(X_std, index=df_x.index, columns=df_x.columns)

print(df_x_std)

# エルボー法

distortions = []
K = range(1,15)
for k in K:
  kmeanModel = KMeans(n_clusters=k).fit(df_x_std)
  kmeanModel.fit(df_x_std)
  distortions.append(sum(np.min(cdist(df_x_std, kmeanModel.cluster_centers_, 'euclidean'), axis=1)) / df_x_std.shape[0])

# # Plot the elbow

fig2, ax2 = plt.subplots()
ax2.plot(K, distortions, 'bx-')
ax2.set_xlabel('k')
ax2.set_ylabel('Distortion')
ax2.set_title('The Elbow Method showing the optimal k')
st.pyplot(fig2)

st.write('適切なクラスター数を判断する材料としてエルボー法のグラフを表示')
st.write('ただし本アプリではクラスター数は６に固定')


# モデルの定義
n_clusters = 6
kmeans = KMeans(n_clusters=n_clusters, random_state=0)

# モデルの学習
kmeans.fit(df_x_std)

# クラスターの中心座標の確認
# kmeans.cluster_centers_

# クラスの格納（ターゲットの横に格納）

# クラスタリングの適用
cluster = kmeans.predict(df_x_std)

df_t.loc[df_x_std.index, ['class']] = cluster

# 各クラスのDFを作成し、リストに格納

target_each_class = []
for i in range(n_clusters):
  d = df_t[df_t['class'] == i]
  target_each_class.append(d)

# 各クラスの結果の平均をリストに格納し、streamlitで表示するDataFrameを作成する

result_list = []
for i in range(n_clusters):
  l = []
  for j in range(len(df_t.columns[:-1])):
    l.append(target_each_class[i].mean()[j]*100)
  l.append(len(target_each_class[i]))
  result_list.append(l)

#result_df = pd.DataFrame(data= result_list, index = [f'Class{x:02d}' for x in range(6)], columns = df_t.columns)
result_df = pd.DataFrame(data= result_list, index = [f'Class{x:02d}' for x in range(6)], 
                         columns = ['１日後のリターン平均',
                                     '３日後のリターン平均',
                                     '５日後のリターン平均',
                                     '１０日後のリターン平均',
                                     'サンプル数'])

markdown = """

## 各クラスターの１日後、３日後、５日後、１０日後の平均リターン
"""
st.markdown(markdown)

st.table(result_df)

# 本日のクラスを計算する

todays_std = sc.fit_transform(df.iloc[-3:, 5:-4])
todays_class = kmeans.predict(todays_std[-1:])

markdown2 = """

## 直近の株価データのクラスターを計算
"""
st.markdown(markdown2)

i = int(todays_class)


#st.subheader(f'本日{}のクラスは{i}です')
st.subheader(f"本日（{df.index[-1:].strftime('%Y-%m-%d')[0]}）のクラスは{i}です")

# 本日のクラスの過去データを取得

todays_df = target_each_class[int(todays_class)]
#print(todays_df)

# 本日のクラスの過去データをｘとｔに切り分ける（ターゲットは５日後の上下）

todays_df_x = df_x_std.loc[todays_df.index, :]
todays_df_t = todays_df.loc[:, '5day_result']

# LightGBMの計算

# データフレームを綺麗に出力する関数
#import IPython
#def display(*dfs, head=True):
#    for df in dfs:
#        IPython.display.display(df.head() if head else df)

# 説明変数,目的変数
X = todays_df_x
y = todays_df_t

# トレーニングデータ,テストデータの分割
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.20, random_state=2)

# モデルの学習
model = lgb.LGBMRegressor() # モデルのインスタンスの作成
model.fit(X_train, y_train) # モデルの学習

# 全データの予測
y_pred = model.predict(X)

# 真値と予測値の表示
#df_pred = pd.DataFrame({'5days return':y_test,'5days return pred':y_pred})
#display(df_pred)

# 散布図を描画(真値 vs 予測値)
#plt.plot(y_test, y_test, color = 'red', label = 'x=y') # 直線y = x (真値と予測値が同じ場合は直線状に点がプロットされる)
#plt.scatter(y_test, y_pred) # 散布図のプロット
#plt.xlabel('y') # x軸ラベル
#plt.ylabel('y_test') # y軸ラベル
#plt.title('y vs y_pred') # グラフタイトル

# モデル評価
# rmse : 平均二乗誤差の平方根
#mse = mean_squared_error(y_test, y_pred) # MSE(平均二乗誤差)の算出
#rmse = np.sqrt(mse) # RSME = √MSEの算出
#print('RMSE :',rmse)

# r2 : 決定係数
#r2 = r2_score(y_test,y_pred)
#print('R2 :',r2)

# 予測値がプラスの時だけトレードする（sim_dfシミュレーション用DFを作って計算）

sim_df = pd.DataFrame(y)
sim_df['pred'] = y_pred
pred_trade_plus = sim_df[sim_df['pred']>0]['pred'].mean()
pred_trade_plus_number = len(sim_df[sim_df['pred']>0]['pred'])
pred_trade_minus = sim_df[sim_df['pred']<=0]['pred'].mean()
pred_trade_minus_number = len(sim_df[sim_df['pred']<=0]['pred'])

st.subheader("""


５日後の株価をLightGBMで予測する

  """)

st.write(f' クラス{todays_class}の５日後のリターンの平均: 　　　　　{y.mean()*100:+.2f}％　{len(y)}個')
st.write(f' クラス{todays_class}の５日後のリターンの予測平均: 　　　{y_pred.mean()*100:+.2f}％　{len(y_pred)}個')
st.write(f' 回帰でプラスの時だけトレードした場合の平均:　{pred_trade_plus*100:+.2f}％　{pred_trade_plus_number}個')
st.write(f' 回帰でマイナスの時だけトレードした場合の平均:　{pred_trade_minus*100:+.2f}％　{pred_trade_minus_number}個')
st.write(f' 本日の回帰予測値:                     　　　 {float(y_pred[-1:])*100:+.2f}%')

st.caption('データは２０１５年１月から直近まで')


st.subheader('アプリ作成のコンセプト')

markdown1 = '''
#### １、株価や経済指標から特徴量を作成して予測を繰り返したが、上手くいかず・・・

[効率的市場仮説](https://www.nomura.co.jp/terms/japan/ko/A02426.html)
　＝　株価の動きはランダムで予測はできない
    
という学説通りだな、と行き詰まる・・・


'''
st.markdown(markdown1)

markdown2 = '''
#### ２、株価の予測をサポートする、という観点から考え直す

自分がトレードするときに考えること

（１）今の株式市場を取り巻く環境は？

    下げやすいか、上げやすいか、荒れた動きか、静かな動きか、など
    
（２）今の環境の中で一番儲かりやすそうな戦略は？

    順張りか、逆張りか、長めの期間でトレードするか、短期勝負か、など

（３）実際にトレードしてみて、微調整

    実際の値動きを見ながら市場のとらえ方を調整

上記のようなことを考えるのはマーケット環境によって同じファクターが違う形で
影響するため

例えば上昇中の株式市場では悪いニュースもそれを克服した後のことにフォーカスがあたり
さらなる上昇の材料になることがある

逆に下落中の株式市場では悪いニュースはそのまま下落の材料になる

    例えばコロナ流行の初期はコロナの拡大は悪いこととして下落を誘発

    しかし流行の末期にはコロナ拡大も集団免疫を作る過程のものなので
    むしろ早期解決が近づいたと株価上昇の材料に

そういったことを考えると、マーケット環境によって回帰モデルも違うのではないか、
と考えるようになった
'''

st.markdown(markdown2)

markdown3 = '''
#### ３、ということでクラスタリングでマーケット環境を分類

株価のトレンドを見るために「移動平均」、株価の直近の変動率を見るために「ATR」、
株価の直近の相対的な位置を見るために「直近高値、安値と終値の関係」を特徴量としてクラスタリング

クラスタリングの結果について各スラスの３日～１０日後の株価リターンの平均を計算してみると、
上昇しているもの、下落しているものなどの傾向がはっきりした

それぞれの傾向があるクラス毎に回帰することで、予測精度は高くなった


'''

st.markdown(markdown3)

st.write('''
         
         ----------------------------------------------------------------
         ''')

st.subheader('〇　問題点、改良すべき点など')

markdown4 = '''
#### オーバーフィッティングの懸念

全銘柄に対応するアプリにするためにtrain dataとtest dataに分けて細かく検証・
調整ができていないため、オーバーフィッティングになったままの可能性がある

#### 日経平均に特化したアプリの作成

１つの銘柄に特化することで、より精緻な分析の上で回帰を行いたい

また分類、回帰と分けなくてもディープラーニング（PyTorch）なら特徴量をうまく設定できれば
良い結果が得られる？

#### 優位性のある取引戦略のブラッシュアップ（やってみて思ったこと）

今回クラスを絞ることで外見上は回帰の精度がよくなった

過去データを使ったシミュレーションなどである程度背景のわかっている
優位性のある戦略を作成し、それを機械学習を利用してブラッシュアップするという
可能性があると思った
（ホワイトボックスで作った戦略をブラックボックスでブラッシュアップする）



'''

st.markdown(markdown4)