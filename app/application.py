# from flask import Flask, render_template, request, jsonify
# # from model_loader_file import load_model_h5
# from keras.preprocessing.text import Tokenizer
# from keras.preprocessing.sequence import pad_sequences
# from sklearn.preprocessing import OneHotEncoder
# import numpy as np
# import pandas as pd
# from keras.models import load_model
# from flask_cors import CORS
# import json
# import joblib
# from sqlalchemy import create_engine

# app = Flask(__name__)
# CORS(app)

# # Load model saat aplikasi dimulai
# model_path = '../model/my_model.h5'
# model = load_model(model_path)
# print("Model loaded successfully")

# # Load Tokenizer untuk preprocessing data
# tokenizer_qualification = Tokenizer(num_words=1000, oov_token="<OOV>")
# tokenizer_skill = Tokenizer(num_words=1000, oov_token="<OOV>")

# # Set OneHotEncoder
# encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
# # Muat encoder yang telah di-fit
# encoder = joblib.load('../encoder_fit/encoder.joblib')


# # Getting database

# def get_event_data_from_database():
#     # Buat koneksi ke database
#     engine = create_engine(
#         'postgresql://postgres:adminch2ps374@34.128.99.62:5432/impacter_db?')

#     # Query data dari tabel event
#     # query = 'SELECT * FROM "Event"'
#     query = 'SELECT "eventId",name,"category" FROM PUBLIC."Category" INNER JOIN PUBLIC."Event" ON "categoryId" = "category_id";'
#     # query_e = 'Select * FROM "Event"'
#     # query_c = 'Select * FROM "Category'

#     # Baca data ke DataFrame
#     df = pd.read_sql_query(query, engine)
#     # df_e = pd.read_sql_query(query_e, engine)
#     # df_c = pd.read_sql_query(query_c, engine)
#     df = df.sample(frac=1).reset_index(drop=True)

#     return df


# # Fungsi untuk melakukan prediksi
# def predict_interest(data):
#     try:
#         data_df = pd.DataFrame(data, index=[0])
#         # Tokenisasi dan padding data
#         data_df['Location'] = data_df['Location'].astype(str)
#         data_df['Type of Organization'] = data_df['Type of Organization'].astype(
#             str)
#         data_df['Skills'] = data_df['Skills'].astype(str)
#         print("Hellow owrdsfasd")
#         print(type(data))
#         print(data)
#         print(data['Location'])
#         print(data['Type of Organization'])
#         print(data['Skills'])
#         # data = data.astype(str)
#         # datas = json.loads(data)
#         # datas = datas.astype(str)
#         # print(datas['Location'])
#         # print(datas['Type of Organization'])
#         # print(datas['Skills'])
#         # data = str(data)
#         location = str(data['Location'])
#         Skills = str(data['Skills'])
#         type_of_organization = str(data['Type of Organization'])
#         # print(data)
#         # print(type(data))
#         # print(data['Location'])
#         # print(data['Type of Organization'])
#         # print(data['Skills'])
#         loc_org = []
#         loc_org.append(location)
#         loc_org.append(type_of_organization)
#         location_org_df = pd.DataFrame(loc_org)
#         print(data_df['Skills'].values[0])
#         print(loc_org, location_org_df)
#         if location is None or type_of_organization is None:
#             print(
#                 "Error: Location or Organization is None")
#             return [1, 2, 2, 22]
#         else:
#             user_loc_org_encoded = encoder.transform(
#                 data_df[['Location', 'Type of Organization']])
#             print(user_loc_org_encoded)
#         print("=============")

#         if Skills is None:
#             print(
#                 "Error: Skill is None")
#             return [1, 2, 2, 22]
#         else:
#             print(data_df['Skills'].values[0])
#             tokenizer_skill.fit_on_texts(data_df['Skills'].values[0])
#             skill_seq = tokenizer_skill.texts_to_sequences(
#                 data_df['Skills'].values[0])
#             print(")))))))))))))")

#             print(type(skill_seq))
#             print(skill_seq)

#             skill_pad = pad_sequences(skill_seq, maxlen=120,
#                                       padding='post', truncating='post')
#         print("+++++++++++++")

#         # memberikan input untuk 'Qualifications' dan 'Category' dan 'Location' dari event.
#         # menggunakan rata-rata atau median dari data training untuk ini.
#         # Memuat nilai rata-rata dari file numpy
#        # Memuat nilai rata-rata dari file numpy
#         average_qualifications_pad = np.load(
#             '../average/average_qualifications_pad.npy')
#         average_event_cat_loc_org_encoded = np.load(
#             '../average/average_event_cat_loc_org_encoded.npy')
#         print(type(average_qualifications_pad))
#         print(type(average_event_cat_loc_org_encoded))
#         print(average_event_cat_loc_org_encoded, average_qualifications_pad)
#         # Cek jika nilai rata-rata tidak None
#         if average_qualifications_pad is not None or average_event_cat_loc_org_encoded is not None:
#             print("1111111111111111")

#             # Lakukan prediksi menggunakan model
#             if skill_pad is None or user_loc_org_encoded is None or average_event_cat_loc_org_encoded is None or average_qualifications_pad is None:
#                 print(
#                     "Error: average_qualifications_pad or average_event_cat_loc_org_encoded is None")
#                 print("222222222222222")
#                 return ["Error semua "]
#             else:

#                 print("3333333333333333")
#                 # skill_seq_np = np.array(skill_pad)
#                 # print(skill_seq_np, skill_seq_np.shape, len(skill_seq_np))
#                 # skill_seq_np_t = np.transpose(skill_seq_np)
#                 # skill_seq_np_t = skill_seq_np_t[0].reshape(1, -1)
#                 # print(skill_seq_np_t, skill_seq_np_t.shape)
#                 print(skill_pad, len(skill_pad), type(
#                     skill_pad), skill_pad.shape)
#                 skill_pad_t = np.transpose(skill_pad)
#                 skill_pad_np_t = skill_pad_t[0].reshape(1, -1)
#                 skill_pad_np_t = pad_sequences(skill_pad_np_t, maxlen=120)
#                 print("skillllllll pad np t")
#                 print(skill_pad_np_t, skill_pad_np_t.shape)
#                 print(user_loc_org_encoded, len(user_loc_org_encoded))
#                 print(average_qualifications_pad, len(
#                     average_qualifications_pad), average_qualifications_pad.shape)
#                 print(average_event_cat_loc_org_encoded,
#                       len(average_event_cat_loc_org_encoded))
#                 user_input = [
#                     skill_pad_np_t,
#                     user_loc_org_encoded,
#                     average_qualifications_pad,
#                     average_event_cat_loc_org_encoded
#                 ]
#                 result = model.predict(user_input)
#                 print("444444444444444")

#                 # Ensure 'result' is a list
#                 # result_list = result.flatten().tolist()

#                 # Change prediksi menjadi DataFrame
#                 predictions_df = pd.DataFrame(
#                     result.flatten(), columns=['Interest_Score'])

#                 # Taking data event from database

#                 event_data = get_event_data_from_database()

#                 # Gabungkan prediksi dengan data event
#                 results_df = pd.concat(
#                     [event_data.reset_index(drop=True), predictions_df], axis=1)

#                 # Menambahkan 'Type of Organization' user ke DataFrame hasil
#                 results_df['User_Organization_Type'] = data_df.iloc[0]['Type of Organization']
#                 # Membuat kolom baru 'Interest_Score_Adjusted' yang memberikan bobot lebih tinggi
#                 # untuk event yang 'Category'-nya cocok dengan 'Type of Organization' user
#                 results_df['Interest_Score_Adjusted'] = np.where(results_df['category'] == results_df['User_Organization_Type'],
#                                                                  results_df['Interest_Score'] * 1.1,
#                                                                  results_df['Interest_Score'])

#                 # sort event berdasarkan 'Interest_Score'
#                 results_df = results_df.sort_values(
#                     by='Interest_Score_Adjusted', ascending=False)

#                 # results_df = results_df.sort_values(
#                 #     by='Interest_Score', ascending=False)

#                 # Tampilkan 5 rekomendasi tertinggi
#                 top_5_recommendations = results_df.head(5)
#                 top_5_recommendations_json = top_5_recommendations.to_json()
#                 print(predictions_df)
#                 print(results_df)
#                 print(top_5_recommendations_json)

#                 # result_listed = json.dumps(result_list)
#                 # result_listed = json.dumps(top_5_recommendations)
#                 result_listed = json.dumps(top_5_recommendations_json)
#                 return result_listed
#                 # return top_5_recommendations_json
#         else:
#             print(
#                 "Error: average_qualifications_pad or average_event_cat_loc_org_encoded is None")
#             return [1, 2, 2, 22]

#         # user_input = [
#         #     skill_pad,
#         #     np.array([[data['Location'], data['Type of Organization']]]),
#         #     qualification_pad,
#         #     np.array([[data['Category'], data['Location']]])
#         # ]
#         # result = model.predict(user_input)

#         # predictions_df = pd.DataFrame(result.flatten(), columns=['Interest_Score'])

#         # # Menggabungkan prediksi dengan data event
#         # results_df = pd.concat([event_test.reset_index(drop=True), predictions_df], axis=1)

#         # # Menambahkan 'Type of Organization' user ke DataFrame hasil
#         # results_df['User_Organization_Type'] = new_userData.iloc[0]['Type of Organization']

#         # # Membuat kolom baru 'Interest_Score_Adjusted' yang memberikan bobot lebih tinggi
#         # # untuk event yang 'Category'-nya cocok dengan 'Type of Organization' user
#         # results_df['Interest_Score_Adjusted'] = np.where(results_df['Category'] == results_df['User_Organization_Type'],
#         #                                                 results_df['Interest_Score'] * 1.1,
#         #                                                 results_df['Interest_Score'])

#         # # Mengurutkan event berdasarkan 'Interest_Score_Adjusted' dalam urutan menurun
#         # results_df = results_df.sort_values(by='Interest_Score_Adjusted', ascending=False)

#         # # Menampilkan 5 rekomendasi tertinggi
#         # top_recommendations = results_df.head(5)
#         # top_recommendations

#         # Ensure 'result' is a list
#         # result_list = result.flatten().tolist()
#         # return result_list
#     except Exception as e:
#         print("Error in predict_interest:", str(e))
#         return [1, 1, 1, 1, 1, 12, 1]


# @app.route('/')
# def home():
#     return render_template('index.html')


# # @app.route('/predict', methods=['POST'])
# # def predict():
# #     # Ambil data dari form atau permintaan AJAX
# #     data = {
# #         'Skills': request.form['Skills'],
# #         'Location': request.form['Location'],
# #         'Type of Organization': request.form['Type_of_Organization'],
# #         # 'Qualifications': request.form['Qualifications'],
# #         # 'Category': request.form['Category']
# #     }
# #     print("Received data:", data)

# #     # Lakukan prediksi menggunakan fungsi predict_interest
# #     result = predict_interest(data)

# #     # Kirim hasil prediksi sebagai JSON
# #     return jsonify({'result': result})


# @app.route('/predict', methods=['GET', 'POST'])
# def predict():
#     try:
#         # Dapatkan data JSON dari permintaan
#         data = request.get_json()

#         # Lakukan prediksi menggunakan fungsi predict_interest
#         result = predict_interest(data)

#         # Kembalikan hasil prediksi sebagai respons JSON
#         return jsonify({'result': result})
#     except Exception as e:
#         # Jika terjadi kesalahan, kembalikan pesan kesalahan
#         return jsonify(error=str(e)), 500


# def predict_interested(data):
#     try:
#         # For testing, stringify the input data
#         result = json.dumps(data)
#         return result
#     except Exception as e:
#         print("Error in predict_interest:", str(e))
#         return []


# if __name__ == '__main__':
#     app.run(debug=True)
