# MPCount-ONNX-Sample
[MPCount](https://github.com/Shimmer93/MPCount)のONNX変換/推論のサンプルです。<br>
変換自体を試したい方はColaboratoryなどで[MPCount-Convert2ONNX.ipynb](MPCount-Convert2ONNX.ipynb)を使用ください。<br>

https://github.com/user-attachments/assets/1c327e97-ba06-4ff9-84b9-bf21e432da58

# Requirement
* OpenCV 4.5.3.56 or later
* onnxruntime-gpu 1.9.0 or later <br>※onnxruntimeでも動作しますが、推論時間がかかるのでGPUをお勧めします

# Demo
以下の何れかのonnxファイルのうち、使用したいものをダウンロードして model に格納してください。
* [MPCount_sta.onnx](https://github.com/Kazuhito00/MPCount-ONNX-Sample/releases/download/v0.0.1/MPCount_sta.onnx)
* [MPCount_stb.onnx](https://github.com/Kazuhito00/MPCount-ONNX-Sample/releases/download/v0.0.1/MPCount_stb.onnx)
* [MPCount_qnrf.onnx](https://github.com/Kazuhito00/MPCount-ONNX-Sample/releases/download/v0.0.1/MPCount_qnrf.onnx)

デモの実行方法は以下です。
```bash
python sample_onnx.py --movie=test.mp4
```
* --device<br>
カメラデバイス番号の指定<br>
デフォルト：0
* --movie<br>
動画ファイルの指定 ※指定時はカメラデバイスより優先<br>
デフォルト：指定なし
* --image<br>
画像ファイルの指定 ※指定時はカメラデバイスや動画より優先<br>
デフォルト：指定なし
* --width<br>
カメラキャプチャ時の横幅<br>
デフォルト：960
* --height<br>
カメラキャプチャ時の縦幅<br>
デフォルト：540
* --model<br>
ロードするモデルの格納パス<br>
デフォルト：model/MPCount_qnrf.onnx

# Reference
* [Shimmer93/MPCount](https://github.com/Shimmer93/MPCount)

# Author
高橋かずひと(https://twitter.com/KzhtTkhs)
 
# License 
MPCount-ONNX-Sample is under [Apache 2.0 License](LICENSE).

# License(Movie, Image)
サンプル動画は[NHKクリエイティブ・ライブラリー](https://www.nhk.or.jp/archives/creative/)の[ロンドン市内 雑踏](https://www2.nhk.or.jp/archives/creative/material/view.cgi?m=D0002050318_00000)を使用しています。<br>
サンプルの画像は[ぱくたそ](https://www.pakutaso.com/)様の[渋谷マークシティの自由通路の様子](https://www.pakutaso.com/20240529145post-51375.html)を使用しています。
