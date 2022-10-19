import cv2
from flask import Flask, Blueprint, render_template, redirect, url_for, request, flash, Response
from utils.config import Config

app = Flask(__name__)
app.config.from_object(Config)


def get_cameras():
    return {'TOP': 0, 'SIDE': 1}


@app.route('/index', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if request.form['btn_control'] == "btn_evolve":
            print("btn_evolve click")
            return redirect(url_for('information'))
        if request.form['btn_control'] == "btn_add":
            print("btn_add click")
            return redirect(url_for('add_product'))
        if request.form['btn_control'] == "btn_change":
            print("btn_change click")
            # return redirect(url_for('index'))
    return render_template('index.html')


@app.route('/information', methods=['GET', 'POST'])
def information():
    return render_template('information.html')


@app.route('/camera_top/<string:camera_id>/', methods=["GET"])
def camera_top(camera_id):
    return Response(get_frame(camera_id),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/camera_side/<string:camera_id>/', methods=["GET"])
def camera_side(camera_id):
    return Response(get_frame(camera_id),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/add_product')
def add_product():
    return render_template('add_product.html', cameras=get_cameras())


def get_frame(camera_id):
    video = cv2.VideoCapture(int(camera_id), cv2.CAP_DSHOW)
    video.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    video.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
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
    # TODO Add Authentication with Flask-Login
    error = None
    if request.method == 'POST':
        if request.form['username'] != 'admin' or request.form['user_password'] != 'admin':
            error = 'Invalid Credentials. Please try again.'
        else:
            return redirect(url_for('index'))
    return render_template('login.html', error=error)


if __name__ == '__main__':
    app.run(debug=True)
