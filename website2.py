from flask import Flask, request, render_template, redirect, url_for
from ultralytics import YOLO
import easyocr
from PIL import Image
import numpy as np
import os
import re
import base64
from io import BytesIO
import pymysql

def is_stolen_vehicle(plate_number):
    conn = pymysql.connect(
    host='localhost',
    user='root',
    password='',        
    db='smart_eye_npr',
    charset='utf8mb4',
    cursorclass=pymysql.cursors.DictCursor
)
    try:
        with conn.cursor() as cursor:
            sql = "SELECT id FROM stolen_vehicles WHERE plate_number = %s"
            cursor.execute(sql, (plate_number,))
            return bool(cursor.fetchone())
    finally:
        conn.close()

app = Flask(__name__)
UPLOAD_FOLDER = 'C:/Users/ishit/OneDrive/Desktop/projects/cv project/static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

model = YOLO('C:/Users/ishit/OneDrive/Desktop/projects/cv project/runs/detect/plate_detection_augmented/weights/best.pt')
ocr_reader = easyocr.Reader(['en'])

PLATE_REGEX = r'^([A-Z]{2}\d{2}[A-Z]{1,2}\d{4}|[0-9]{2}BH[0-9]{4}[A-Z]{2})$'


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'camera_image' in request.form and request.form['camera_image']:
            img_data = request.form['camera_image']
            img_data = img_data.split(',')[1]
            img = Image.open(BytesIO(base64.b64decode(img_data)))
            filename = 'capture.png'
            filepath = os.path.join(UPLOAD_FOLDER, filename)
            img.save(filepath)
            current_filename = filename
        # --- REGULAR FILE UPLOAD ---
        elif 'image' in request.files:
            file = request.files['image']
            if file.filename == '':
                return 'No selected file', 400
            filepath = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(filepath)
            current_filename = file.filename
        else:
            return 'No input', 400

        # --- Plate Detection and OCR ---
        results = model(filepath)
        boxes = results[0].boxes
        if len(boxes) == 0:
            plate_text = "No number plate detected."
        else:
            box = max(boxes, key=lambda b: (b.xyxy[0][2] - b.xyxy[0][0]) * (b.xyxy[0][3] - b.xyxy[0][1]))
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            orig_img = Image.open(filepath)
            plate_img = orig_img.crop((x1, y1, x2, y2))

            ocr_result = ocr_reader.readtext(np.array(plate_img))
            raw_ocr_texts = [re.sub(r'[^A-Za-z0-9]', '', res[1].upper()) for res in ocr_result]
            possible_plates = set()
            for i in range(len(raw_ocr_texts)):
                for j in range(i, len(raw_ocr_texts)):
                    combined = ''.join(raw_ocr_texts[i:j+1])
                    if re.fullmatch(PLATE_REGEX, combined):
                        possible_plates.add(combined)
            all_joined = ''.join(raw_ocr_texts)
            if re.fullmatch(PLATE_REGEX, all_joined):
                possible_plates.add(all_joined)

            if possible_plates:
                detected_plate = list(possible_plates)[0]
                if is_stolen_vehicle(detected_plate):
                    plate_text = f"Detected Plate Number: {detected_plate}<br><span class='text-danger fw-bold'>⚠ Reported Stolen Vehicle ⚠</span>"
                else:
                    plate_text = f"Detected Plate Number: {detected_plate}<br><span class='text-success fw-bold'>Not Reported Stolen</span>"
            else:
                plate_text = "OCR Results: " + ", ".join([res[1] for res in ocr_result]) if ocr_result else "Could not read plate text."




        return render_template('result.html', image_url=current_filename, plate_text=plate_text)

    return render_template('upload.html')

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return redirect(url_for('static', filename='uploads/' + filename), code=301)

if __name__ == '__main__':
    app.run(debug=True)
