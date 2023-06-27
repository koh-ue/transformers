# Visions

これは, 自分のカメラで撮った画像から, 画像分類を行うモデルを生成するためのプログラムです.

**※現状, 学習したモデルが読み込めなくなってしまっております.**

## Quick Start
---

1. Pythonの実行環境を構築します. Pythonのバージョンはどんなものでも構いませんが, 後でPython 3.8を使わなければいけないタイミングがやってきます. 必要なpackageは全て, requirements.txtに入っているので, 
   ```
   pip install -r requirements.txt
   ```
   とすることで, 一括でインストールできます.
2. まずは、学習したモデルを授業のときと同じように、zipファイルで保存しましょう. 
3. 保存したzipファイルから、run_vision_transformer_own_data.pyを使って、学習モデルを構築します. これにより、zipファイルと同じディレクトリに, model.pth(PyTorchモデル)と、params.json(実験で使ったパラメータ)が生成されます.
   ```
   python run_vision_transformer_own_data.py --zipfile <自分のzipファイルのパス>
   ```
   例:
   ```
   python run_vision_transformer_own_data.py --zipfile ../result/data/archive.zip
   ```
4. 次に、model.pthを、汎用形式であるONNXに変換します. これにより, model.onnxが生成されます(付随して, simplified.onnxも生成されます).
   ```
   python torch2onnx.py --model <自身の.pthモデルのパス>
   ```
   例:
   ```
   python torch2onnx.py --model ../result/data/model.pth
   ```
5. 最後に, ONNXモデルを, KPUモデル(.kmodel形式)に変換します. このプログラムは、nncaseというパッケージを利用しており, _**Ubuntu 20.04, Python 3.8 上でしか動かない**_ ようです. よって, お好みの環境で実行するか, Dockerで仮想環境を構築しましょう. (参考: [https://github.com/kendryte/nncase/blob/master/docs/USAGE_EN.md](https://github.com/kendryte/nncase/blob/master/docs/USAGE_EN.md))
   ```
   python onnx2kmodel.py --model <自分のonnxモデルのパス>
   ```
   例:
   ```
   python onnx2kmodel.py --model ../result/data/model.onnx
   ```