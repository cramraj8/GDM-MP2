from collections import defaultdict
import numpy as np


ANSWER_LOG_STATS_FILE = "./answerSequenceStats.txt"
QUESTION_LOG_STATS_FILE = "./questionSequenceStats.txt"
LOG_WRITE_FILE = "./aggregatedSequenceStats.txt"

Q_TYPES = ["what", "where", "how", "when", "who", "why"]


def main():
	answer_length_dict = defaultdict(list)
	answer_length_list = []
	with open(ANSWER_LOG_STATS_FILE, 'r') as f:
		answer_data = f.read().strip().split("\n")
		for answer_line in answer_data:
			if answer_line == "":
				continue
			tmp_list = answer_line.split("-")
			if len(tmp_list) < 2:
				continue
			q_type = tmp_list[-2]
			try:
				seq_length = int(tmp_list[-1])
			except Exception:
				continue
			answer_length_dict[q_type].append(seq_length)
			answer_length_list.append(seq_length)

			# break

	with open(LOG_WRITE_FILE, 'a') as f:
		f.write("Answer : Total sequence stats        : max-{}\tmin-{}\tmean-{}\n".format(max(answer_length_list), min(answer_length_list), np.mean(answer_length_list)))
		for q_type in Q_TYPES:
			f.write("Answer : {} sequence stats : max-{}\tmin-{}\tmean-{}\n".format(q_type, max(answer_length_dict[q_type]), min(answer_length_dict[q_type]), np.mean(answer_length_dict[q_type])))
		f.write("\n\n")


	question_length_dict = defaultdict(list)
	question_length_list = []
	with open(QUESTION_LOG_STATS_FILE, 'r') as f:
		question_data = f.read().strip().split("\n")
		for question_line in question_data:
			if question_line == "":
				continue
			tmp_list = question_line.split("-")
			if len(tmp_list) < 2:
				continue
			q_type = tmp_list[-2]
			try:
				seq_length = int(tmp_list[-1])
			except Exception:
				continue
			question_length_dict[q_type].append(seq_length)
			question_length_list.append(seq_length)

			# break


	with open(LOG_WRITE_FILE, 'a') as f:
		f.write("Question : Total sequence stats        : max-{}\tmin-{}\tmean-{}\n".format(max(question_length_list), min(question_length_list), np.mean(question_length_list)))
		for q_type in Q_TYPES:
			f.write("Question : {} sequence stats : max-{}\tmin-{}\tmean-{}\n".format(q_type, max(question_length_dict[q_type]), min(question_length_dict[q_type]), np.mean(question_length_dict[q_type])))
		f.write("\n\n")



if __name__ == "__main__":
	main()