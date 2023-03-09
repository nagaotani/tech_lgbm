import streamlit as st

st.subheader('アプリ作成のコンセプト')

markdown1 = '''
#### １、株価や経済指標から特徴量を作成してPyTorchなどで予測を繰り返したが、上手くいかず・・・

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
'''

st.markdown(markdown2)

markdown3 = '''
#### ３、ということでクラスタリングでマーケット環境を分類

クラスタリングして各スラスの３日～１０日後の株価リターンの平均を計算してみると、
上昇しているもの、下落しているものなどの傾向がはっきりした

それぞれの傾向があるクラス毎に回帰することで、予測精度は高くなった


'''

st.markdown(markdown3)

st.subheader('〇　問題点、改良すべき点など')

markdown4 = '''
#### オーバーフィッティングの懸念

全銘柄に対応するアプリにするためにtrain dataとtest dataに分けての
テストができていないため、オーバーフィッティングになっている可能性がある

#### 日経平均に特化したアプリの作成

１つの銘柄に特化することで、より精緻な分析の上で回帰を行いたい

#### 優位性のある取引戦略のブラッシュアップ

今回クラスを絞ることで外見上は回帰の精度がよくなった

過去データを使ったシミュレーションなどである程度背景のわかっている
優位性のある戦略を作成し、それを機械学習を利用してブラッシュアップするという
可能性があると思った
（ホワイトボックスで作った戦略をブラックボックスでブラッシュアップする）



'''

st.markdown(markdown4)