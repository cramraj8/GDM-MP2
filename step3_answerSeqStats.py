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
LOG_WRITE_FILE = "./answerSequenceStats.txt"
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

	total_ans_cnt = 0
	simple_ans_found_cnt = 0
	nltk_ans_found_cnt = 0

	simple_seq_list = []
	nltk_seq_list = []
	for sample_img, sample_ans in zip(sg_data, qa_data):
		log_write_string = ""
		if sample_img['image_id'] != sample_ans['id']:
			print("IDs did not match !")
			IDs_mismatched_counts += 1
			continue
		
		# ======================================================
		# 					Answer
		# ======================================================
		n_qas = len(sample_ans['qas'])
		total_ans_cnt += n_qas # UPDATED
		for qa in sample_ans['qas']:
			tmp_qa = qa['answer'].replace(".", "").lower().split(" ") # process the answer sequence
			q_type = qa['question'].split(" ")[0]
			total_q_type_cnt[q_type] += 1 # UPDATED

			tmp_answer_seq = nltk.word_tokenize(qa['answer'].replace(".", "").lower())
			answer_token_lemma = [ lmtzr.lemmatize(token) for token in tmp_answer_seq ]
			
			# ==================================================
			# 					Simple tokenized stats
			# ==================================================
			simple_seq_list.append(tmp_qa)

			# ==================================================
			# 					NLTK tokenized stats
			# ==================================================
			nltk_seq_list.append(answer_token_lemma)

			with open(LOG_WRITE_FILE, 'a') as f:
				f.write("{}-{}-".format(sample_img['image_id'], qa['qa_id']))
				f.write(".".join(answer_token_lemma))
				# f.write("-")
				# f.write(".".join(answer_token_lemma))
				f.write("-{}".format(q_type.lower()))
				f.write("-{}".format(len(answer_token_lemma)))
				f.write("\n")


		samples_cnt += 1
		print("Finished (image, qas) pair - {} / {}".format(samples_cnt, n_samples))

		# break

	with open(LOG_WRITE_FILE, 'a') as f:
		f.write("\n\nCompletion info - \n\t mismatched img & qa IDs count : {}".format(IDs_mismatched_counts))


if __name__ == "__main__":
	main()
