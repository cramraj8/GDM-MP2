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

WRITE_INTERVAL = 10000
SRC_SG_PATH = "../VG-data/scene_graphs.json"
SRC_QA_FILE = "../VG-data/question_answers.json"
LOG_WRITE_FILE = "./results_questionChecker.txt"
LOG_SEMANTIC_PAIR_FILE = "./semantic_pairs_questionChecker.csv"
Q_TYPES = ["What", "Where", "How", "When", "Who", "Why"]
# SIM_THRESHOLD = 0.9
# SIM_THRESHOLD2 = 0.5
SIM_THRESHOLD = 0.325


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
	semantic_q_type_found_cnt = defaultdict(int)

	total_ques_cnt = 0
	simple_ques_found_cnt = 0
	nltk_ques_found_cnt = 0
	semantic_ques_found_cnt = 0	

	semantic_pairs = []
	for sample_img, sample_ques in zip(sg_data, qa_data):
		log_write_string = ""
		if sample_img['image_id'] != sample_ques['id']:
			print("IDs did not match !")
			IDs_mismatched_counts += 1
			continue
			
		# ======================================================
		# 					Scene-graph 
		# ======================================================
		sg_img_objs = sample_img['objects'] # todo: take relationships into list as well, besides entities
		obj_name_list = []
		for sg_img_obj in sg_img_objs:
			obj_names = sg_img_obj['names']
			if len(obj_names) == 1:
				obj_name = obj_names[0].replace(".", "").lower().split(" ") # process the question sequence
			obj_name_list.append(obj_name)
		
		obj_name_list = list(set(itertools.chain.from_iterable(obj_name_list))) # todo: lemmantize
		
		# ======================================================
		# 					question
		# ======================================================
		n_qas = len(sample_ques['qas'])
		total_ques_cnt += n_qas # UPDATED
		# simple_found_questions_cnt = 0
		# nltk_found_questions_cnt = 0
		# sim_found_questions_cnt = 0
		for qa in sample_ques['qas']:
			tmp_qa = qa['question'].replace(".", "").lower().split(" ") # process the question sequence
			q_type = qa['question'].split(" ")[0]
			total_q_type_cnt[q_type] += 1 # UPDATED

			tmp_question_seq = nltk.word_tokenize(qa['question'])
			question_token_lemma = [ lmtzr.lemmatize(token).replace(".", "").lower() for token in tmp_question_seq ]
			
			# ==================================================
			# 					Simple tokenized stats
			# ==================================================
			for ques_token in tmp_qa:
				if ques_token in obj_name_list:
					simple_ques_found_cnt += 1 # UPDATED
					simple_q_type_found_cnt[q_type] += 1 # UPDATED
					break

			# ==================================================
			# 					NLTK tokenized stats
			# ==================================================
			for q_token in question_token_lemma[1:]:
				if q_token in obj_name_list:
					nltk_ques_found_cnt += 1 # UPDATED
					nltk_q_type_found_cnt[q_type] += 1 # UPDATED
					break

			# ==================================================
			# 					Semantic similarity stats
			# ==================================================
			max_sim_dict = {}
			for sg_token in obj_name_list:
				for q_token in question_token_lemma[1:]: # [1:] --> excludes 'what' token          
					try:
						tmp_sim = model.similarity(q_token, sg_token)
						max_sim_dict[tmp_sim] = [q_token, sg_token, tmp_sim]
					except KeyError:
						continue
							
			if len(max_sim_dict) == 0:
				empty_sim_list_qas_cnt += 1	 
				continue  
			max_sim_val = max(max_sim_dict.keys())        
			if max_sim_val >= SIM_THRESHOLD:
				semantic_q_type_found_cnt[q_type] += 1 # UPDATED
				semantic_ques_found_cnt += 1 # UPDATED
				semantic_pairs.append(max_sim_dict[max_sim_val])


		samples_cnt += 1
		print("Finished (image, qas) pair - {} / {} : where question found in SG - {} / {}".\
			  format(samples_cnt, n_samples, simple_ques_found_cnt, total_ques_cnt))

		if ((samples_cnt == 1) or (samples_cnt % WRITE_INTERVAL == 0)):

			with open(LOG_WRITE_FILE, 'w') as f:
				f.write("sample ID: {}\n\t - Finished (img, qas) pair \t\t- {} / {} \n".format(sample_img['image_id'], samples_cnt, n_samples))

				f.write("\t - Question type-level stats :")
				# ========= simple tokenizer stats
				f.write("\n\t\t(simple tokenizer)  : {}/{} : ".format(simple_ques_found_cnt, total_ques_cnt))
				for q_type_name in Q_TYPES:
					f.write(" {} - {}/{}".format(q_type, simple_q_type_found_cnt[q_type_name], total_q_type_cnt[q_type_name]))
				# ========= nltk normalizer stats`
				f.write("\n\t\t(nltk normalizer)   : {}/{} : ".format(nltk_ques_found_cnt, total_ques_cnt))
				for q_type_name in Q_TYPES:
					f.write(" {} - {}/{}".format(q_type, nltk_q_type_found_cnt[q_type_name], total_q_type_cnt[q_type_name]))
				# ========= semantic normalizer stats
				f.write("\n\t\t(semantic sim)      : {}/{} : ".format(semantic_ques_found_cnt, total_ques_cnt))
				for q_type_name in Q_TYPES:
					f.write(" {} - {}/{}".format(q_type, semantic_q_type_found_cnt[q_type_name], total_q_type_cnt[q_type_name]))

			df = pd.DataFrame(semantic_pairs)
			df.to_csv(LOG_SEMANTIC_PAIR_FILE, index=True)

		# print(simple_q_type_found_cnt)
		# print(nltk_q_type_found_cnt)
		# print(semantic_q_type_found_cnt)

		# break

	with open(LOG_WRITE_FILE, 'a') as f:
		f.write("\n\nCompletion info - \n\t mismatched img & qa IDs count : {}".format(IDs_mismatched_counts))
		f.write("\n\t qas count that found empty similarity list : {}".format(empty_sim_list_qas_cnt))



if __name__ == "__main__":
	main()
