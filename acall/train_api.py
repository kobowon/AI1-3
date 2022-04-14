from flask import Flask, request, jsonify
import json
import sys
from train_json import train 
app = Flask (__name__)
 
    

#데코레이더로 라우팅 경로 지정
@app.route('/acall/test')
def test():
    print('b')
    print('a', file=sys.stderr)
    app.logger.info('a')
    app.logger.error('b')
    app.logger.info('c')
    return "acall test api b"



@app.route('/acall/trainer', methods = ['POST'])
def train_acall():
    data = request.get_json()
    with open('./data/acall_data_simple2.json', 'w', encoding='utf-8') as f:
        json.dump(data,f) 
    print(data)
    train('./data/acall_data_simple2.json')
    print(type(data)) #dict
    
    
    return json.dumps(data, ensure_ascii=False) #한글을 출력할 때 에러가 있을 수 있어서 jsonify 대신 json.dumps(string) 씀, json string 반환 

   

 
if __name__ == "__main__":
    app.run(host="0.0.0.0", port="8087", debug=True)

