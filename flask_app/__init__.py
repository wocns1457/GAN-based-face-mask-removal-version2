# /app.py
from models import Mask_G, Face_G
from prediction import *
from flask import Flask, render_template, request, flash
from werkzeug.utils import secure_filename


INPUT_FILE = ''
PREDICTION_FILE = ''
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

mask, face = Mask_G(filters=32), Face_G(filters=64)
mask_checkpoint_dir = './mask32_checkpoints'
face_checkpoint_dir = './face_checkpoints'
model = Load_model(mask, face, mask_checkpoint_dir=mask_checkpoint_dir, face_checkpoint_dir=face_checkpoint_dir)
model.load()

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
           
def create_app(config=None):
    app = Flask(__name__)
    app.secret_key = 'format'
    
    @app.route('/',methods=('GET', 'POST')) # 접속하는 url
    def main():
        return render_template('main.html')


    @app.route('/view', methods = ['GET', 'POST'])
    def view():
        if request.method == 'POST':
            f = request.files['file']
            
            if allowed_file(f.filename):
                f.save('flask_app/static/image/' + secure_filename(f.filename))
                INPUT_FILE = 'image/' + f.filename
                img = "flask_app/static/image/"+f.filename
                pred_img = 'prediction/' + f.filename
                pred(img, model.mask_model, model.face_model)
                return render_template('view.html', input_img=INPUT_FILE, pred_img=pred_img)
            
            elif f.filename == '':
                flash('파일을 선택해 주세요.')
                return render_template('main.html')
            
            else:
                flash('이미지 파일의 형식이 아닙니다.')
                return render_template('main.html')
            
    return app
app = create_app()