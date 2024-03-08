from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin
import os
import requests
from FlagEmbedding import BGEM3FlagModel
# from sentence_transformers import util


app = Flask(__name__)
CORS(app, support_credentials=True)

# Initialize BGEM3FlagModel outside the route functions
model = BGEM3FlagModel('BAAI/bge-m3', use_fp16=True)


@app.route('/compute_score', methods=['POST'])
@cross_origin(supports_credentials=True)
def cosine_score():
    try:
        # Extract sentences from the request
        sentences_1 = request.form.get('sent1')
        sentences_2 = request.form.get('sent2')
        # print(sentences_1, sentences_2)
        # Encode sentences to obtain dense vectors
        embeddings_1 = model.encode(sentences_1, batch_size=12, max_length=8192)['dense_vecs']
        embeddings_2 = model.encode(sentences_2)['dense_vecs']
        
        # Compute similarity matrix
        similarity = embeddings_1 @ embeddings_2.T
        similarity = round(similarity * 100, 2) 
        # Compute cosine similarity between the first embeddings of each sentence
        # cosine_similarity = (embeddings_1[0] @ embeddings_2[0].T).item()
        # print(similarity, cosine_similarity)
        # cosine_similarity = util.cos_sim(embeddings_1[0], embeddings_2[0])
        # print("Cosine Similarity between the first embeddings:")
        # print(cosine_similarity)
        return jsonify({'similarity_matrix': similarity.tolist()}), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500

