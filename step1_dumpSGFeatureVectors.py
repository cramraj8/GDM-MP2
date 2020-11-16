import os
import h5py
import nltk
import spacy
import gensim
import en_core_web_sm
from nltk.data import find
from nltk.corpus import wordnet
import numpy as np
import json
import collections
import copy

SRC_SG_PATH = "../VG-data/scene_graphs.json"
SRC_QA_FILE = "../VG-data/question_answers.json"

word2vec_sample = str(find('models/word2vec_sample/pruned.word2vec.txt'))
model = gensim.models.KeyedVectors.load_word2vec_format(word2vec_sample, binary=False)
lmtzr = nltk.WordNetLemmatizer()

DST_SG_FEATURES_DATA_FOLDER = "./intermediate_files/sg_features/"


if not os.path.exists(DST_SG_FEATURES_DATA_FOLDER):
    os.makedirs(DST_SG_FEATURES_DATA_FOLDER)


sg_data = json.load(open(SRC_SG_PATH, 'r'))
print(len(sg_data)) # num-images

qa_data = json.load(open(SRC_QA_FILE, 'r'))
print(len(qa_data)) # num-qa (== num-img)


def main():

	n_img_data = len(sg_data)
	for img_idx, sample_img in enumerate(sg_data):
	    img_id = sample_img['image_id']
	    
	    save_feature_file = os.path.join(DST_SG_FEATURES_DATA_FOLDER, "{}.h5".format(img_id))
	    with h5py.File(save_feature_file, 'w') as hf:
	        
	        obj_features_list = []
	        for obj in sample_img['objects']:
	            obj_name = obj['names'][0]
	            obj_name_seq = nltk.word_tokenize(obj_name.lower())
	            obj_name_token_lemma = [ lmtzr.lemmatize(token) for token in obj_name_seq ]
	            emb_matrix = []
	            for token in obj_name_token_lemma:
	                try:
	                    tmp_vec = model[token]
	                    emb_matrix.append(tmp_vec)
	                except KeyError:
	                    continue
	            if emb_matrix == []:
	                emb_matrix = np.random.normal(0, 1, size=(300,))
	            emb_matrix = np.asarray(emb_matrix, np.float32)
	            semantic_feature = np.mean(emb_matrix, axis=0) # feature

	            # save_feature_file = os.path.join(DST_SG_FEATURES_DATA_FOLDER, "{}.h5".format(obj['object_id']))
	            # with h5py.File(save_feature_file, 'w') as hf:
	            #     hf.create_dataset("features", data=semantic_feature)
	            # hf.create_dataset(str(obj['object_id']), data=semantic_feature, chunks=True, compression="gzip", compression_opts=9)
	            hf.create_dataset(str(obj['object_id']), data=semantic_feature)
	           
	    if img_idx % 100 == 0: print("Finished extracting features : {}/{}".format(img_idx + 1, n_img_data))
	    if img_idx == 1000: break

if __name__ == "__main__":
	main()
