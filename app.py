from flask import Flask, request, render_template
import inference_app

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Rendering the results from the HTML Graphics User Interface
    sent = request.form.get('sentence')
    # Make predictions on the given sentence
    pred_ = inference_app.get_sentiment(sent=sent)
    return render_template('index.html', prediction = 'This is a {} sentence!'.format(pred_))


if __name__ == '__main__':
    app.run(debug=True)