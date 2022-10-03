import cv2
from flask import Flask, Blueprint, render_template, redirect, url_for, request, flash, Response
from utils.config import Config

app = Flask(__name__)
app.config.from_object(Config)


@app.route('/index', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if request.form['btn_control'] == "btn_evolve":
            print("btn_evolve click")
            return redirect(url_for('information'))
        if request.form['btn_control'] == "btn_add":
            print("btn_add click")
            return redirect(url_for('video_feed'))
        if request.form['btn_control'] == "btn_change":
            print("btn_change click")
            # return redirect(url_for('index'))
    return render_template('index.html')


@app.route('/information', methods=['GET', 'POST'])
def information():
    return render_template('information.html')


@app.route('/video_feed')
def video_feed():
    video = cv2.VideoCapture(0)
    return Response(gen(video),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


def gen(video):
    while True:
        ret, frame = video.read()
        if not ret:
            video.release()
            break
        ret, png = cv2.imencode('.png', frame)
        frame = png.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')


@app.route("/")
@app.route('/login', methods=['GET', 'POST'])
def login():
    error = None
    if request.method == 'POST':
        if request.form['username'] != 'admin' or request.form['user_password'] != 'admin':
            error = 'Invalid Credentials. Please try again.'
        else:
            return redirect(url_for('index'))
    return render_template('login.html', error=error)


if __name__ == '__main__':
    app.run(debug=True)
