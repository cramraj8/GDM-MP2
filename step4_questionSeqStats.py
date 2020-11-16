import json
import itertools
from collections import defaultdict
from nltk.corpus import wordnet
import pandas as pd
import nltk
import spacy
import gensim
import en_core_web_sm
from nltk.data import find

word2vec_sample = str(find('models/word2vec_sample/pruned.word2vec.txt'))
model = gensim.models.KeyedVectors.load_word2vec_format(word2vec_sample, binary=False)
lmtzr = nltk.WordNetLemmatizer()
nlp = en_core_web_sm.load()

# WRITE_INTERVAL = 10000
SRC_SG_PATH = "../VG-data/scene_graphs.json"
SRC_QA_FILE = "../VG-data/question_answers.json"
# LOG_WRITE_FILE = "./results_answerChecker.txt"
# LOG_SEMANTIC_PAIR_FILE = "./semantic_pairs_answerChecker.csv"
LOG_WRITE_FILE = "./questionSequenceStats.txt"
Q_TYPES = ["What", "Where", "How", "When", "Who", "Why"]
# SIM_THRESHOLD = 0.9
# SIM_THRESHOLD2 = 0.5
# SIM_THRESHOLD = 0.325


def main():	
	# ==========================================================================================
	# 					Load Data
	# ==========================================================================================
	sg_data = json.load(open(SRC_SG_PATH, 'r'))
	print("Total image samples : ", len(sg_data)) # num-images

	qa_data = json.load(open(SRC_QA_FILE, 'r'))
	print("Total QA samples : ", len(qa_data)) # num-qa (== num-img)

	# ==========================================================================================
	# 					Handle QA data
	# ==========================================================================================
	n_samples = len(sg_data)
	samples_cnt = 0

	IDs_mismatched_counts = 0
	empty_sim_list_qas_cnt = 0

	total_q_type_cnt = defaultdict(int)
	simple_q_type_found_cnt = defaultdict(int)
	nltk_q_type_found_cnt = defaultdict(int)

	total_ques_cnt = 0
	simple_ques_found_cnt = 0
	nltk_ques_found_cnt = 0

	simple_seq_list = []
	nltk_seq_list = []
	for sample_img, sample_ques in zip(sg_data, qa_data):
		log_write_string = ""
		if sample_img['image_id'] != sample_ques['id']:
			print("IDs did not match !")
			IDs_mismatched_counts += 1
			continue
		
		# ======================================================
		# 					question
		# ======================================================
		n_qas = len(sample_ques['qas'])
		total_ques_cnt += n_qas # UPDATED
		for qa in sample_ques['qas']:
			tmp_qa = qa['question'].replace(".", "").lower().split(" ") # process the question sequence
			q_type = qa['question'].split(" ")[0]
			total_q_type_cnt[q_type] += 1 # UPDATED

			tmp_question_seq = nltk.word_tokenize(qa['question'].replace(".", "").lower())
			question_token_lemma = [ lmtzr.lemmatize(token) for token in tmp_question_seq ]
			
			# ==================================================
			# 					Simple tokenized stats
			# ==================================================
			simple_seq_list.append(tmp_qa)

			# ==================================================
			# 					NLTK tokenized stats
			# ==================================================
			nltk_seq_list.append(question_token_lemma)

			with open(LOG_WRITE_FILE, 'a') as f:
				f.write("{}-{}-".format(sample_img['image_id'], qa['qa_id']))
				f.write(".".join(question_token_lemma))
				# f.write("-")
				# f.write(".".join(question_token_lemma))
				f.write("-{}".format(q_type.lower()))
				f.write("-{}".format(len(question_token_lemma)))
				f.write("\n")


		samples_cnt += 1
		print("Finished (image, qas) pair - {} / {}".format(samples_cnt, n_samples))

		# break

	with open(LOG_WRITE_FILE, 'a') as f:
		f.write("\n\nCompletion info - \n\t mismatched img & qa IDs count : {}".format(IDs_mismatched_counts))


if __name__ == "__main__":
	main()
