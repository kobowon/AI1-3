from flask import Flask, request, jsonify
import json
import sys
from train_json import train 
from infer_json import infer
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



@app.route('/acall/train', methods = ['POST'])
def train_acall():
    data = request.get_json() #학습 데이터 dict 형식으로 전달
    train_file = './data/acall_data_simple2.json' #파일로 제작
    with open(train_file, 'w', encoding='utf-8') as f:
        json.dump(data,f) 
    page_num = len(data['data']) #acall page 수 (분류할 대상, 즉 라벨 수)
    print(f'분류할 페이지 수는 {page_num} 입니다')

    train(train_file, page_num) #checkpoint file & inference를 위한 config.json 저장    
    return 'train complete'


@app.route('/acall/inference', methods = ['POST'])
def infer_acall():
    data = request.get_json() #추론 데이터 dict 형식으로 전달
    print(data)
    questions = data['questions']
    answer = infer(text_list = questions, batch_size = 10)
    result = {"pages" : answer} 
    print(f"질문들에 대한 슈퍼앱 페이지 번호는 {answer} 입니다")
    
    #한글을 출력할 때 에러가 있을 수 있어서 jsonify 대신 json.dumps(string) 씀, json string 반환 
    return json.dumps(result, ensure_ascii=False)
    

if __name__ == "__main__":
    app.run(host="0.0.0.0", port="8087", debug=True)

