# 이 폴더 위치에서 pip install flask를 실행하면 flask를 설치할 수 있습니다.
from flask import Flask, jsonify, request, render_template
import joblib

app = Flask(__name__)

@app.route('/iris')
def iris():
  sepalLength = float(request.args.get('sepalLength'))
  sepalWidth = float(request.args.get('sepalWidth'))
  petalLength = float(request.args.get('petalLength'))
  petalWidth = float(request.args.get('petalWidth'))
  data = [sepalLength, sepalWidth, petalLength, petalWidth]

  #model from h5
  model = joblib.load('./rf_iris.h5')
  result = model.predict([data]) # []가 두개가 됨
  print(result)

  return jsonify({'result': result[0][5:] })

def home(): # hjtml 파일 실행했을 경우 > 지금은 안 쓸것
  return render_template('iris.html')

if __name__ == '__main__':
  app.run(host='127.0.0.1', port=5000, debug=True)