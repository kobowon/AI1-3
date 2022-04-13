from flask import Flask, request, jsonify
app = Flask (__name__)
 
    

#데코레이더로 라우팅 경로 지정
@app.route('/acall/test')
def test():
    return "acall test api"


@app.route('/acall/pageMap', methods = ['POST'])
def classify():
    param = request.get_json()
    print(param)
    return jsonify(param)
 
if __name__ == "__main__":
    app.run(host="0.0.0.0", port="8088")

