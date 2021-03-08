from flask import Flask,render_template,request
import pickle
import numpy as np

app = Flask(__name__)

model=pickle.load(open('model.pkl','rb'))

@app.route('/')
def hello_world():
    return render_template("diabetes.html")



@app.route('/predict',methods=['POST','GET'])
def predict():
    int_features=[int(x) for x in request.form.values()]
    final=[np.array(int_features)]
    print(int_features)
    print(final)
    prediction=model.predict(final)
    output='{0:.{1}f}'.format(prediction[0], 2)

    if output > str(0.5):
        return render_template('diabetes.html',
                               pred='You are diabetic person.\nProbability of diabetes is {}'.format(output),
                               bhai="kuch karna hain iska ab?")
    else:
        return render_template('diabetes.html',
                               pred='You are not diabetic person. \n Probability of being diabetic is {}'.format(output),
                               bhai="Your Forest is Safe for now")



if __name__ == '__main__':
    app.run(debug=True)




