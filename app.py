from flask import Flask, request, render_template, jsonify, url_for, session, g
import cv2
import numpy as np
import os
import sys
from EigenFace import generate_new_face
import base64
from flask_mysqldb import MySQL

app = Flask(__name__, static_folder='static')
app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = 'devyani45'
app.config['MYSQL_DB'] = 'ppllist'

mysql = MySQL(app)
username = "admin"
password = "admin123"

from EigenFace import load_images, generate_data_matrix

NUM_EIGEN_FACES = 10
MAX_SLIDER_VALUE = 255

@app.route('/')
def home():
    return render_template('home.html')
@app.route('/login')
def login():
    return render_template('login.html')
@app.route('/admin')
def admin():
    return render_template('admin.html')
@app.route('/faces')
def faces():
    return render_template('faces.html')
@app.route('/match')
def match():
    return render_template('match.html')
@app.route('/get_table_data', methods=['POST'])
@app.route('/get_table_data', methods=['POST'])
def get_table_data():
    data = request.get_json()
    table_name = data.get('table_name')
    
    if not table_name:
        return jsonify({'error': 'Table name is required'}), 400
    
    cursor = mysql.connection.cursor()
    cursor.execute(f"SELECT * FROM {table_name}")
    rows = cursor.fetchall()
    column_names = [i[0] for i in cursor.description]

    table_data = [dict(zip(column_names, row)) for row in rows]

    return jsonify(table_data)
@app.route('/insert_data', methods=['POST'])
def insert_data():
    data = request.get_json()
    table_name = data.get('table_name')
    values = data.get('values')

    if not table_name or not values:
        return jsonify({'error': 'Table name and values are required'}), 400

    cursor = mysql.connection.cursor()
    try:
        columns = ', '.join(values.keys())
        placeholders = ', '.join(['%s'] * len(values))
        query = f"INSERT INTO {table_name} ({columns}) VALUES ({placeholders})"
        cursor.execute(query, tuple(values.values()))
        mysql.connection.commit()
        return jsonify({'success': 'Data inserted successfully'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        cursor.close()
@app.route('/get_table_columns', methods=['POST'])
def get_table_columns():
    data = request.get_json()
    table_name = data.get('table_name')

    if not table_name:
        return jsonify({'error': 'Table name is required'}), 400

    cursor = mysql.connection.cursor()
    cursor.execute(f"SHOW COLUMNS FROM {table_name}")
    columns = [column[0] for column in cursor.fetchall()]

    return jsonify(columns)
@app.route('/delete_record', methods=['POST'])
def delete_record():
    data = request.get_json()
    table_name = data.get('table_name')
    id_value = data.get('id')

    if not table_name or not id_value:
        return jsonify({'error': 'Table name and ID are required'}), 400

    cursor = mysql.connection.cursor()
    try:
        query = f"DELETE FROM {table_name} WHERE s_no = %s"
        cursor.execute(query, (id_value,))
        mysql.connection.commit()
        return jsonify({'success': 'Record deleted successfully'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        cursor.close()


@app.route('/index')
def index():
    return render_template('index.html', NUM_EIGEN_FACES=NUM_EIGEN_FACES, MAX_SLIDER_VALUE=MAX_SLIDER_VALUE)

@app.route('/update_result', methods=['POST'])
def update_result():
    slider_values = [int(request.form.get(f'weight{i}')) for i in range(NUM_EIGEN_FACES)]
    
    result_face = generate_new_face(slider_values)  

    _, buffer = cv2.imencode('.png', result_face)
    result_image_base64 = base64.b64encode(buffer).decode('utf-8')

    return jsonify({'result_image': result_image_base64})


@app.route('/process_images', methods=['POST'])
def process_images():
    try:
        images_dir = "images"
        image_list = load_images(images_dir)
        img_size = image_list[0].shape
        data_matrix = generate_data_matrix(image_list)
        
        print("Performing PCA ", end="...")
        mean, eigen_vectors = cv2.PCACompute(data_matrix, mean=None, maxComponents=NUM_EIGEN_FACES)
        print("DONE")
        
        mean_face = mean.reshape(img_size)
        eigen_faces = [vector.reshape(img_size) for vector in eigen_vectors]
        
        return jsonify({'message': 'Processing Complete'})
    except cv2.error as e:
        print(f"OpenCV error: {e}")
        return jsonify({'error': str(e)}), 500
    except Exception as e:
        print(f"General error: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
