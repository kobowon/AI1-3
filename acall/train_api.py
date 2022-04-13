from flask import Flask, request, jsonify
import json
app = Flask (__name__)
 
    

#데코레이더로 라우팅 경로 지정
@app.route('/acall/test')
def test():
    return "acall test api"


@app.route('/acall/trainer', methods = ['POST'])
def train():
    data = request.get_json()
    with open('./data/acall_data_simple2.json', 'w', encoding='utf-8') as f:
        json.dump(data,f) 
    print(param)
    return json.dumps(param)
 
if __name__ == "__main__":
    app.run(host="0.0.0.0", port="8087")

