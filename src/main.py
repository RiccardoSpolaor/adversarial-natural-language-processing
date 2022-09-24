
import itertools
import pickle, os, sys

print(sys.path)
#sys.path.append(sys.path.append(os.path.dirname(os.path.abspath(__file__))))

#print(sys.path.append(os.path.dirname(os.path.abspath(__file__))))

from utils.seed_setter import set_seed
from utils.black_box import BlackBox
from utils.attacker import Attacker

set_seed()

with open(os.path.join('./pickle_data/train_test_data/test_data.pickle'), 'rb') as f:
    x_test, y_test = pickle.load(f)
f.close()

black_box = BlackBox()

all_preds = black_box.predict_all(x_test)



attack_list = []

for sent, lab, pred in list(zip(x_test, y_test, all_preds)):
        if round(pred) == lab and len(sent) < 500:
            attack_list += [(sent,lab)]

print('Accuracy for the attack list:')
black_box.evaluate([sent for sent, lab in attack_list], [lab for sent, lab in attack_list])

attacker = Attacker()

max_words = [5, 10, 20]
top_neighbours = [10, 20, 30]
top_lm = [4, 8]
max_children = [2, 4]
max_pop_members = [10, 20]

combinations = list(itertools.product(*[max_words, top_neighbours, top_lm, max_children, max_pop_members]))

BATCH_SIZE = 2 # 100

os.makedirs(os.path.join('./pickle_data/experiments'), exist_ok=True)

for idx_comb, comb in enumerate(combinations[:2]):
    print('##########################################################')
    print('####################COMBINATION {}/{}:####################'.format(idx_comb+1, len(combinations)))
    print('##########################################################\n')

    adversarial_list = []
    time_list = []
    generations_list = []

    for i in range(BATCH_SIZE):
        print('####################ATTACK {}/{}:####################'.format(i+1, BATCH_SIZE))
        new_review, tot_time, tot_gens = attacker.attack(attack_list[i][0], attack_list[i][1], max_words=comb[0], top_neighbours=comb[1], top_lm=comb[2], max_children=comb[3], max_pop_members=comb[4])
        if new_review is None:
            new_review = attack_list[i][0]
        
        adversarial_list += [(new_review, attack_list[i][1])]
        time_list += [tot_time]
        generations_list += [tot_gens]

    evaluation = black_box.evaluate([sent for sent, lab in adversarial_list], [lab for sent, lab in adversarial_list])

    description = ('Parameters:\n'
                    '\tMaximum words to try to change per population member: {}\n'
                    '\tMaximum top neighbours to choose per word to change: {}\n' 
                    '\tMaximum top language model substitutes to choose per word to change: {}\n'
                    '\tMaximum children per generation: {}\n'
                    '\tMaximum population members: {}\n').format(*comb)

    with open(os.path.join('./pickle_data/experiments/maxwords{}_topneighbours{}_toplm{}_maxchildren{}_maxpopmembers{}.pickle'.format(*comb)), 'wb') as f:
        pickle.dump(
            {
                'description': description,
                'max_words' : comb[0], 
                'top_neighbours' : comb[1], 
                'top_lm' : comb[2], 
                'max_children' : comb[3], 
                'max_pop_members' : comb[4],
                'adversarial_list': adversarial_list, 
                'time_list': time_list, 
                'generations_list': generations_list, 
                'evaluation': evaluation[1]
            }, f)
    f.close()
