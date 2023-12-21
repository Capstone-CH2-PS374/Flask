from flask import Flask, render_template, request, jsonify
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import pandas as pd
from keras.models import load_model
from flask_cors import CORS
import requests
import json
import os
import joblib
from sqlalchemy import create_engine

app = Flask(__name__)
CORS(app)

# Load model saat aplikasi dimulai
model_path = './model/my_model.h5'
model = load_model(model_path)
print("Model loaded successfully")

# Load Tokenizer untuk preprocessing data
tokenizer_qualification = Tokenizer(num_words=1000, oov_token="<OOV>")
tokenizer_skill = Tokenizer(num_words=1000, oov_token="<OOV>")

# Set OneHotEncoder
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
# Muat encoder yang telah di-fit
encoder = joblib.load('./encoder_fit/encoder.joblib')


# Getting database

def get_event_data_from_database():
    # Buat koneksi ke database
    engine = create_engine(
        'postgresql://postgres:adminch2ps374@34.128.99.62:5432/impacter_db?')

    query = 'SELECT "eventId",name,"category" FROM PUBLIC."Category" INNER JOIN PUBLIC."Event" ON "categoryId" = "category_id";'
    # query_IT = 'SELECT * FROM PUBLIC."Category" WHERE "category" = "IT";'
    # Baca data ke DataFrame
    df = pd.read_sql_query(query, engine)
    # df_it = pd.read_sql_query(query_IT, engine)
    df = df.sample(frac=1).reset_index(drop=True)

    return df


# Fungsi untuk melakukan prediksi
def predict_interest(data):
    try:
        data_df = pd.DataFrame(data, index=[0])
        # Tokenisasi dan padding data
        data_df['Location'] = data_df['Location'].astype(str)
        data_df['Type of Organization'] = data_df['Type of Organization'].astype(
            str)
        data_df['Skills'] = data_df['Skills'].astype(str)
        location = str(data['Location'])
        Skills = str(data['Skills'])
        type_of_organization = str(data['Type of Organization'])

        loc_org = []
        loc_org.append(location)
        loc_org.append(type_of_organization)
        location_org_df = pd.DataFrame(loc_org)

        user_loc_org_encoded = encoder.transform(
            data_df[['Location', 'Type of Organization']])
        tokenizer_skill.fit_on_texts(data_df['Skills'].values[0])
        skill_seq = tokenizer_skill.texts_to_sequences(
            data_df['Skills'].values[0])
        skill_pad = pad_sequences(skill_seq, maxlen=120,
                                  padding='post', truncating='post')

        # memberikan input untuk 'Qualifications' dan 'Category' dan 'Location' dari event.
        # menggunakan rata-rata atau median dari data training untuk ini.
        # Memuat nilai rata-rata dari file numpy
       # Memuat nilai rata-rata dari file numpy
        average_qualifications_pad = np.load(
            './average/average_qualifications_pad.npy')
        average_event_cat_loc_org_encoded = np.load(
            './average/average_event_cat_loc_org_encoded.npy')

        # Transpose skill_pad for make input shape to model
        skill_pad_t = np.transpose(skill_pad)
        skill_pad_np_t = skill_pad_t[0].reshape(1, -1)
        skill_pad_np_t = pad_sequences(skill_pad_np_t, maxlen=120)

        # Input model
        user_input = [
            skill_pad_np_t,
            user_loc_org_encoded,
            average_qualifications_pad,
            average_event_cat_loc_org_encoded
        ]
        result = model.predict(user_input)

        # Change prediksi menjadi DataFrame
        predictions_df = pd.DataFrame(
            result.flatten(), columns=['Interest_Score'])

        # Taking data event from database
        event_data = get_event_data_from_database()
        # print(event_data)
        # if event_data['category'] == 'IT':
        #     print(event_data['category'])
        # Gabungkan prediksi dengan data event
        results_df = pd.concat(
            [event_data.reset_index(drop=True), predictions_df], axis=1)

        # Menambahkan 'Type of Organization' user ke DataFrame hasil
        results_df['User_Organization_Type'] = data_df.iloc[0]['Type of Organization']
        # Membuat kolom baru 'Interest_Score_Adjusted' yang memberikan bobot lebih tinggi
        # untuk event yang 'Category'-nya cocok dengan 'Type of Organization' user
        results_df['Interest_Score_Adjusted'] = np.where(results_df['category'] == results_df['User_Organization_Type'],
                                                         results_df['Interest_Score'] * 1.5,
                                                         results_df['Interest_Score']
                                                         )
        results_df['Interest_Score_Adjusted'] = np.where(results_df['eventId'].notna(),
                                                         results_df['Interest_Score'] * 1.3,
                                                         results_df['Interest_Score']
                                                         )

        # sort event berdasarkan 'Interest_Score'
        results_df = results_df.sort_values(
            by='Interest_Score_Adjusted', ascending=False)

        # Tampilkan 5 rekomendasi tertinggi
        top_5_recommendations = results_df.head(5)
        top_5_recommendations_json = top_5_recommendations.to_json()
        print(predictions_df)
        print(results_df)
        print(top_5_recommendations_json)

        result_listed = json.loads(top_5_recommendations_json.replace("\"",'"'))
        json_without_slash = json.dumps(result_listed)
        url = 'http://localhost:3000/events/predict'
        response = requests.post(url, json=top_5_recommendations_json)
        if response.status_code == 200:
            print("Data sent successfully")
        else:
            print("Failed to send data. Status code:", response.status_code)
        return json_without_slash

    except Exception as e:
        print("Error in predict_interest:", str(e))
        return ["Error in predict_interest() "]


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    try:
        # Dapatkan data JSON dari permintaan
        data = request.get_json()

        # Lakukan prediksi menggunakan fungsi predict_interest
        result = predict_interest(data)

        # Kembalikan hasil prediksi sebagai respons JSON
        return jsonify({'result': result})
    except Exception as e:
        # Jika terjadi kesalahan, kembalikan pesan kesalahan
        return jsonify(error=str(e)), 500


def predict_interested(data):
    try:
        # For testing, stringify the input data
        result = json.dumps(data)
        return result
    except Exception as e:
        print("Error in predict_interest:", str(e))
        return []


if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
