import string
import re
import os
import numpy as np
import pickle
import licensed_scripts.lm_1b_eval as google_language_model_utils
import nltk
from utils.black_box import BlackBox
import time

nltk.download('maxent_ne_chunker', quiet = True)
nltk.download('words', quiet = True)

class Attacker(object):
    
    def __init__ (self):
        with open(os.path.join('./pickle_data/attack_utils/tokens_dictionary.pickle'), 'rb') as f:
            tokens_dictionary, inverse_tokens_dictionary = pickle.load(f)
            self.__tokens_dictionary = tokens_dictionary
            self.__inverse_tokens_dictionary = inverse_tokens_dictionary
        f.close()
        
        self.__black_box = BlackBox()
        self.__distance_matrix = np.load(os.path.join('./numpy_files/distance_matrix.npy'))
        self.__google_lm = google_language_model_utils.LM()
        self.__stopwords = nltk.corpus.stopwords.words('english')
        self.__sentence_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
        self.__tree_bank_word_tokenizer = nltk.tokenize.TreebankWordTokenizer()
        
        latin_similar = "’'‘ÆÐƎƏƐƔĲŊŒẞÞǷȜæðǝəɛɣĳŋœĸſßþƿȝĄƁÇĐƊĘĦĮƘŁØƠŞȘŢȚŦŲƯY̨Ƴąɓçđɗęħįƙłøơşșţțŧųưy̨ƴÁÀÂÄǍĂĀÃÅǺĄÆǼǢƁĆĊĈČÇĎḌĐƊÐÉÈĖÊËĚĔĒĘẸƎƏƐĠĜǦĞĢƔáàâäǎăāãåǻąæǽǣɓćċĉčçďḍđɗðéèėêëěĕēęẹǝəɛġĝǧğģɣĤḤĦIÍÌİÎÏǏĬĪĨĮỊĲĴĶƘĹĻŁĽĿʼNŃN̈ŇÑŅŊÓÒÔÖǑŎŌÕŐỌØǾƠŒĥḥħıíìiîïǐĭīĩįịĳĵķƙĸĺļłľŀŉńn̈ňñņŋóòôöǒŏōõőọøǿơœŔŘŖŚŜŠŞȘṢẞŤŢṬŦÞÚÙÛÜǓŬŪŨŰŮŲỤƯẂẀŴẄǷÝỲŶŸȲỸƳŹŻŽẒŕřŗſśŝšşșṣßťţṭŧþúùûüǔŭūũűůųụưẃẁŵẅƿýỳŷÿȳỹƴźżžẓ"
        safe_characters = string.ascii_letters + string.digits + latin_similar + ' '
        safe_characters += "'"
        self.__safe_characters = safe_characters
        
        #self.__max_words = 5 # 5; 10; 20
        #self.__top_neighbours = 8 # 10; 20; 30
        #self.__top_lm = 4 # 5; 8; 10
        self.__MAX_GENS = 2 #20
        #self.__max_children = 2 # 2; 5; 10
        #self.__max_pop_members = 5 # 10; 15; 20
        
        
    def __handle_contractions(self, sentence):
        sentence = self.__tree_bank_word_tokenizer.tokenize(sentence)
        return ' '.join(sentence)
    
    def __preprocess_sentence(self, sentence):
        chars = set(w for w in sentence)
        symbols = [c for c in chars if not c in self.__safe_characters]
        for symbol in symbols:
            new_sentence = sentence.replace(symbol, ' ' + symbol + ' ')
        new_sentence = self.__handle_contractions(new_sentence)
        return new_sentence#.split()
    
    def __split_review(self, review):
        sentences = self.__sentence_tokenizer.tokenize(review)
        labels = [None] * len(sentences)
        for index, sentence in enumerate(sentences):
            sentences[index] = self.__preprocess_sentence(sentence)
            sentences[index] = nltk.ne_chunk(nltk.pos_tag(nltk.word_tokenize(sentences[index])))
            labels[index] = [None] * len(sentences[index])
            for i, chunk in enumerate(sentences[index]):
                if hasattr(chunk, 'label'):
                    labels[index][i] = 1
                    sentences[index][i] = ' '.join(c[0] for c in chunk)
                else:
                    labels[index][i] = 0
                    sentences[index][i] = chunk[0]
        return sentences, labels
            
    def __most_similar(self, word, delta = 0.5, num_words = 20):
        try:
            index = self.__tokens_dictionary[word]
        except:
            return []

        if (index > self.__distance_matrix.shape[0]):
            return []

        dist_order = np.argsort(self.__distance_matrix[index,:])[1:num_words+1]
        dist_list = self.__distance_matrix[index][dist_order]

        mask = np.ones_like(dist_list)
        mask = np.where(dist_list < delta)
        return [self.__inverse_tokens_dictionary[index] for index in dist_order[mask]]         
    
    def __rejoin_review(self, sentences):
        new_sentences = sentences[:]
        for i, sent in enumerate (new_sentences):
            new_sentence = ' '.join(sent)
            new_sentence = re.sub(r' ([^A-Za-z0-9])', r'\1', new_sentence)
            new_sentences[i] = new_sentence
        return ' '.join(new_sentences)
    
    def __perturb(self, top_neighbours, top_lm, sentences, word_index, neighbours, y_target, changed_words_list = []):
        
        #perturbation_start_time = time.time()
        
        prefix = ' '.join(sentences[word_index[0]][ : word_index[1]])
        suffix = ' '.join(sentences[word_index[0]][word_index[1]+1 : -1])
        
        lm_preds = self.__google_lm.get_words_probs(
            prefix, 
            neighbours[ : min(top_neighbours, len(neighbours))], 
            suffix
        )
        
        score_list = []
        for adv_w in np.argsort(lm_preds)[- min(top_lm, len(lm_preds)) : ]: 
            adv_sentences = sentences[:]
            adv_splitted_text = adv_sentences[word_index[0]][:]
            adv_splitted_text[word_index[1]] = neighbours[adv_w]
            adv_sentences[word_index[0]] = adv_splitted_text
            adv_review = self.__rejoin_review(adv_sentences)
            score = self.__black_box.predict_sentiment(adv_review)
            score_list += [(adv_sentences, score, changed_words_list + [word_index])]
            
        adv_reviews_sorted =  sorted(score_list, key=lambda x: x[1])    
        
        #print('%perturbation_time = {} seconds\n'.format(int(time.time() - perturbation_start_time)))
        #print('Final review:')
        
        if y_target == 0:
            #print(self.__rejoin_review(adv_reviews_sorted[0][0]) + '\n')
            #print('Score:' + str(adv_reviews_sorted[0][1]) + '\n')
            return adv_reviews_sorted[0]
            
        else:
            #print(self.__rejoin_review(adv_reviews_sorted[-1][0]) + '\n')
            #print('Score:' + str(adv_reviews_sorted[-1][1]) + '\n')
            return adv_reviews_sorted[-1]
        
        
    def __crossover(self, parent1, parent2):
        parent1_copy = parent1[0]
        parent2_copy = parent2[0]
        changed_word_list = []
        for i in range(len(parent1_copy)):
            for j in range(len(parent1_copy[i])):
                if np.random.uniform() < 0.5:
                    parent1_copy[i][j] = parent2_copy[i][j]
                    if (i,j) in parent2[2]:
                        changed_word_list += [(i,j)]
                elif (i,j) in parent2[2]:
                    changed_word_list += [(i,j)]
        score = self.__black_box.predict_sentiment(self.__rejoin_review(parent1_copy))
        return parent1_copy, score, changed_word_list
    
    def __get_neighbours_dictionary(self, sentences, labels):
        neighbours_dictionary = {}
        for sent_idx, sent in enumerate(sentences):
            for word_idx, word in enumerate(sent):
                if labels[sent_idx][word_idx] == 1:
                    neighbours_dictionary[(sent_idx, word_idx)] = []
                else:
                    neighbours_dictionary[(sent_idx, word_idx)] = self.__most_similar(
                        word = word.lower(), delta = 0.5, num_words = 50
                    )

        return neighbours_dictionary
    
    def __get_words_to_change(self, max_words, neighbours_dictionary, sentences, changed_word_list = []):
        neighbours_length = {key: len(value) for key, value in neighbours_dictionary.items()}

        for key in neighbours_length.keys():
            if sentences[key[0]][key[1]].lower() in self.__stopwords or key in changed_word_list:
                neighbours_length[key] = 0

        length_sum = sum(neighbours_length.values())

        if length_sum == 0:
            return None
        
        neighbours_length = {key: value/length_sum for key, value in neighbours_length.items()}

        probabilities = list(neighbours_length.values())

        random_choice_size = min(len(np.nonzero(probabilities)[0]), max_words)
        
        return np.random.choice(
            len(neighbours_length.keys()), size = random_choice_size, replace = False, p = probabilities
        )
        
    def attack (self, x_orig, y_orig, max_words = 5, top_neighbours = 10, top_lm = 4, max_children = 2, max_pop_members = 10):
        
        print('Parameters:')
        print('\tMaximum words to try to change per population member: {}'.format(max_words))
        print('\tMaximum top neighbours to choose per word to change: {}'.format(top_neighbours))
        print('\tMaximum top language model substitutes to choose per word to change: {}'.format(top_lm))
        print('\tMaximum children per generation: {}'.format(max_children))
        print('\tMaximum population members: {}\n'.format(max_pop_members))

        attack_start_time = time.time()
        
        y_target = int(not y_orig)
        sentences, labels = self.__split_review(x_orig)
        
        neighbours_dictionary = self.__get_neighbours_dictionary(sentences, labels)
        words_to_change = self.__get_words_to_change(max_words, neighbours_dictionary, sentences)
        
        if words_to_change is None:
            total_time = int(time.time() - attack_start_time)
            return None, total_time, 0
        
        print('Original sentence: \n' + x_orig)
        print('Starting Score: {}; Original Label: {}; Target Label: {}\n'.format(
            self.__black_box.predict_sentiment(x_orig), y_orig, y_target
        ))
        
        print('Generating population... \n')
        
        generation_start_time = time.time()

        population = []
        for index in words_to_change:
            #print('Starting perturbation {}/{}...\n'.format(i + 1, len(words_to_change)))
            new_member = self.__perturb(
                top_neighbours,
                top_lm,
                sentences, 
                list(neighbours_dictionary.keys())[index], 
                neighbours_dictionary[list(neighbours_dictionary.keys())[index]], 
                y_target
            )
            
            if round(new_member[1]) == y_target:
                print('ATTACK SUCCESS!!!\n')
                final_sentence = self.__rejoin_review(new_member[0])
                print('Final adversarial sentence:\n {} \n Score: {}\n'.format(
                    self.__rejoin_review(new_member[0]), new_member[1]
                ))
                total_time = int(time.time() - attack_start_time)
                print('%total_time = {} seconds\n\n'.format(total_time))
                return final_sentence, total_time, 0
            
            population += [new_member]

        for i in range(self.__MAX_GENS):

            if len(population) == 0:
                print('ATTACK FAILED...\n')
                total_time = int(time.time() - attack_start_time)
                print('%total_time = {} seconds\n\n'.format(total_time))
                return None, total_time, i + 1
            
            print('Generation #{}: \n'.format(i+1))
            
            sorted_population =  sorted(population, key=lambda x: x[1])    


            if y_target == 0:
                best_attack = sorted_population[0]
                sorted_population = sorted_population[ : min(max_pop_members, len(sorted_population))]

            else:
                best_attack = sorted_population[-1]
                sorted_population = sorted_population[- min(max_pop_members, len(sorted_population)) : ]
                
            #print(len(sorted_population))
                
            print('Best Adversarial:\n {} \n Score: {}\n'.format(self.__rejoin_review(best_attack[0]), best_attack[1]))

            #if round(best_attack[1]) == y_target:
            #    return self.__rejoin_review(best_attack[0])
            
            print('%generation_time = {} seconds\n'.format(int(time.time() - generation_start_time)))
        
            #best_adversarials = [best_attack]
            if y_target == 1:
                pop_scores = np.array([score for review, score, _ in sorted_population])
            else:
                pop_scores = np.array([1 - score for review, score, _ in sorted_population])
                
            logits = np.exp(pop_scores / 0.3)
            selection_probabilities = logits / np.sum(logits)
            
            
            #print('POP SCORES: ' + str(pop_scores))
            #print('PROBS: ' + str(selection_probabilities))
            #if y_target == 0:
            #    selection_probabilities = selection_probabilities[::-1]
                
            parent_list_1 = np.random.choice(
                len(sorted_population), size=len(sorted_population), p=selection_probabilities)
            parent_list_2 = np.random.choice(
                len(sorted_population), size=len(sorted_population), p=selection_probabilities)
            
            children = [self.__crossover(
                sorted_population[parent_list_1[i]], 
                sorted_population[parent_list_2[i]]
            ) for i in range(len(sorted_population))]
            
            sorted_children =  sorted(children, key=lambda x: x[1])    
            
            if y_target == 0:
                #best_attack = sorted_children[0]
                sorted_children = sorted_children[ : min(max_children, len(sorted_children))]

            else:
                #best_attack = sorted_children[-1]
                sorted_children = sorted_children[- min(max_children, len(sorted_children)) : ]
            
            perturbated_children = []
            
            print('Regenerating population... \n')
            
            generation_start_time = time.time()
            
            for child in sorted_children:
                
                sentences = child[0]
                
                neighbours_dictionary = self.__get_neighbours_dictionary(sentences, labels)
                words_to_change = self.__get_words_to_change(max_words, neighbours_dictionary, sentences, child[2])
                
                if words_to_change is None:
                    words_to_change = self.__get_words_to_change(max_words, neighbours_dictionary, sentences)
        
                if words_to_change is None:
                    continue

                for index in words_to_change:
                    #print('Starting perturbation {}/{}:\n'.format(i + 1, len(children) * len(words_to_change)))
                    #i = i+1
                    new_member = self.__perturb(
                        top_neighbours,
                        top_lm,
                        sentences, 
                        list(neighbours_dictionary.keys())[index], 
                        neighbours_dictionary[list(neighbours_dictionary.keys())[index]], 
                        y_target,
                        child[1]
                    )
                    
                    
                    
                    if round(new_member[1]) == y_target:
                        print('ATTACK SUCCESS!!!\n')
                        final_sentence = self.__rejoin_review(new_member[0])
                        print('Final adversarial sentence:\n {} \n Score: {}\n'.format(
                            self.__rejoin_review(new_member[0]), new_member[1]
                        ))

                        total_time = int(time.time() - attack_start_time)
                        print('%total_time = {} seconds\n\n'.format(total_time))
                        return final_sentence, total_time, i

                    perturbated_children += [new_member]

            population = [best_attack] + perturbated_children
        
        print('ATTACK FAILED...\n')
        total_time = int(time.time() - attack_start_time)
        print('%total_time = {} seconds\n\n'.format(total_time))
        return None, total_time, self.__MAX_GENS

