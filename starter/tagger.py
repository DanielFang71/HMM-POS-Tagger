# The tagger.py starter code for CSC384 A4.
# Currently reads in the names of the training files, test file and output file,
# and calls the tagger (which you need to implement)
import sys

import numpy as np


def tag(training_list, test_file, output_file):
    # Tag the words from the untagged input file and write them into the output file.
    # Doesn't do much else beyond that yet.
    print("Tagging the file.")
    #
    # YOUR IMPLEMENTATION GOES HERE
    #

    # To reduce copying the massive arrays around, combine a lot of functions to
    # tag and also reduce the need for global variables.
    print("Start fetching states and observations...")
    train_words, test_words = [], []
    complete_train_lines = []
    first_symbol_lst = []
    tags = set()
    words = set()

    print("Start reading training files...")
    # read training files
    for train_file in training_list:
        f = open(train_file, 'r', encoding='UTF-8')
        line = f.readline().strip()
        first_word, first_symbol = get_word_symbol(line)
        first_symbol_lst.append(first_symbol)
        while line != '':
            complete_train_lines.append(line)
            word, symbol = get_word_symbol(line)
            tup = (word, symbol)
            train_words.append(tup)
            tags.add(symbol)
            words.add(word)
            line = f.readline().strip()
        f.close()

    print("Start reading test file...")
    # read test file
    f = open(test_file, 'r', encoding='UTF-8')
    line = f.readline().strip()
    while line != '':
        test_words.append(line)
        line = f.readline().strip()
    f.close()

    # record index of tags and words to reduce the list.index function use.
    # use list.index inside nested loop is a big waste of time
    tag_dict = {}
    for key, word in enumerate(tags):
        tag_dict[word] = key

    word_dict = {}
    for key, word in enumerate(words):
        word_dict[word] = key

    len_tag = len(tags)
    len_voc = len(words)

    # for adding weight for transition probability
    tag_distribution = np.zeros(len_tag)
    emission_table = np.zeros((len_tag, len_voc))
    transition_table = np.zeros((len_tag, len_tag))

    print("Create initial table and emission table...")
    init = np.zeros(len_tag)
    for i in range(len(complete_train_lines) - 1):
        line = complete_train_lines[i]
        word, symbol = get_word_symbol(line)
        row = tag_dict[symbol]
        col = word_dict[word]
        emission_table[row, col] += 1
        tag_distribution[row] += 1
        if word in [".", "!", ";"]:
            next_line = complete_train_lines[i + 1]
            next_symbol = get_word_symbol(next_line)[1]
            index = tag_dict[next_symbol]
            init[index] += 1

    emission_table = emission_table / emission_table.sum(axis=1, keepdims=True)
    tag_distribution = tag_distribution / len_tag

    for symbol in first_symbol_lst:
        index = tag_dict[symbol]
        init[index] += 1
    init = init / init.sum()

    print("Create transition table...")
    # build transition_table
    for i in range(len_tag):
        curr_symbol = list(tags)[i]
        transit_lst = np.zeros(len_tag)
        create_transit_lst(transit_lst, curr_symbol, complete_train_lines,
                           tag_dict)
        transit_lst = transit_lst / transit_lst.sum()
        transition_table[i] = transit_lst

        # viterbi
    tagged_seq = Viterbi(test_words, tags, init, tag_dict,
                         transition_table, words, word_dict, emission_table,
                         tag_distribution)
    print("Write output...")
    write_output(tagged_seq, output_file)
    print("Done.")


def Viterbi(test_words, tags, init, tag_dict, transition_table, words,
            word_dict,
            emission_table, tag_distribution):
    print("Start viterbi...")
    state = []
    lst_tags = list(tags)

    for key, word in enumerate(test_words):
        print("Viterbi iteration {}/{}".format(key, len(test_words) - 1))
        curr_prob = []
        transition_prob_lst = []
        for tag_i, curr_tag in enumerate(lst_tags):
            if key == 0:
                index = tag_i
                transition_p = init[index]
            else:
                last_state_index = tag_dict[state[-1]]
                transition_p = transition_table[last_state_index][tag_i]

                # compute emission and state probabilities
            if curr_tag not in tags or word not in words:
                emission_p = 0
            else:
                row = tag_i
                col = word_dict[word]
                emission_p = emission_table[row, col]

            state_probability = emission_p * transition_p
            curr_prob.append(state_probability)

            # POS tag occurrence probability
            tag_p = tag_distribution[tag_i]

            # add weight to transition prob with tag occurance probability.
            transition_p = tag_p * transition_p
            transition_prob_lst.append(transition_p)

        curr_max = max(curr_prob)
        # use transition probability for unknown words (p == 0)
        if curr_max == 0:
            curr_max = max(transition_prob_lst)
            state_max = lst_tags[transition_prob_lst.index(curr_max)]
        else:
            state_max = lst_tags[curr_prob.index(curr_max)]
        state.append(state_max)
    print("Finish Viterbi.")
    return list(zip(test_words, state))


def create_transit_lst(transit_lst, target, complete_train_lines, tag_dict):
    for i in range(len(complete_train_lines) - 1):
        curr_line = complete_train_lines[i]
        curr_symbol = get_word_symbol(curr_line)[1]
        if curr_symbol == target:
            next_line = complete_train_lines[i + 1]
            next_symbol = get_word_symbol(next_line)[1]
            index = tag_dict[next_symbol]
            transit_lst[index] += 1


def get_word_symbol(line):
    line_lst = line.split(' : ')
    word, symbol = line_lst[0], line_lst[-1]
    return word, symbol


def write_output(path, output_file):
    # clean output file
    open(output_file, 'w', encoding='UTF-8').close()
    # write to output file
    file = open(output_file, 'w', encoding='UTF-8')
    for tup in path:
        word = tup[0]
        symbol = tup[1]
        file.write(word + ' : ' + symbol + '\n')
    file.close()


if __name__ == '__main__':
    # Run the tagger function.
    print("Starting the tagging process.")

    # Tagger expects the input call: "python3 tagger.py -d <training files> -t <test file> -o <output file>"
    parameters = sys.argv
    training_list = parameters[
                    parameters.index("-d") + 1:parameters.index("-t")]
    test_file = parameters[parameters.index("-t") + 1]
    output_file = parameters[parameters.index("-o") + 1]
    print("Training files: " + str(training_list))
    print("Test file: " + test_file)
    print("Output file: " + output_file)

    # Start the training and tagging operation.
    tag(training_list, test_file, output_file)
