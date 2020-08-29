# generate_jp_name
深層学習による生成モデルです。

LSTMモデルにより、日本人の名前を生成します。

### 学習
```
python train.py
```

### 生成
```
python generate.py
アヤ
スズネ
ナナミ
```

実装はPytorchを使っています。
コードは下記チュートリアルを参考としています。
http://pytorch.org/tutorials/intermediate/char_rnn_generation_tutorial.html
