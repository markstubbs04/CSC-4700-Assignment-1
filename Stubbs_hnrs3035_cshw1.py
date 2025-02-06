import argparse

import random

import pickle

class NGram:
    def __init__(self, n):
        self.num = n
        with open("moby_dick.txt", "r",errors='ignore') as f:
            self.moby_dick = f.read()
    
    def train(self, moby_dick):
        md = str(moby_dick)
        md = md.split()
        word_list = set(md) # convert moby dick to list then to set

        #convert set to dictionary one for unigram and one for bigram
        uni_dict = {word: {} for word in word_list}
        bi_dict = {}
        
        

        if self.num==1:
            # keep track of previous word
            prev_word = "" 

            #iterate through each word in the story
            for word in md: 
                #get the dict for the previous word and increment/add the current word
                if(prev_word != ""):
                    lil_uni_dict = uni_dict[prev_word]

                    #increment value of word in unigram dictionary
                    if word in lil_uni_dict:
                        lil_uni_dict[word] += 1
                    else:
                        lil_uni_dict[word] = 1
                
                
                prev_word = word
                
            #Turn current count based stats into percents
            for word in uni_dict:
                lil_uni_dict = uni_dict[word]
                total = 0
                #get total word count
                for key in lil_uni_dict:
                    total += lil_uni_dict[key]
                #turn original count into a percent
                for key in lil_uni_dict:
                    lil_uni_dict[key] = (lil_uni_dict[key] / total) * 100
            
            return uni_dict

        elif self.num==2:
            # keep track of previous words
            prev_word = md[1] 
            pprev_word = md[0]
            flag = False
            count = 0
            for word in md: 
                if flag: #check if first/second word
                    #get the dict for the previous word and increment/add the current word
                    if (pprev_word, prev_word) not in bi_dict:
                        bi_dict[(pprev_word,prev_word)] = {}
                    lil_bi_dict = bi_dict[(pprev_word,prev_word)]
                    
                    #increment value of word in bigram dictionary
                    if word in lil_bi_dict:
                        lil_bi_dict[word] += 1 
                    else:
                        lil_bi_dict[word] = 1
                    
                    #
                    pprev_word = prev_word
                    prev_word = word
                else:
                    if count == 1:
                        flag = True
                    count += 1

            for word in bi_dict:
                lil_bi_dict = bi_dict[word]
                total = 0
                #get total word count
                for key in lil_bi_dict:
                    total += lil_bi_dict[key]
                #turn original count into a percent
                for key in lil_bi_dict:
                    lil_bi_dict[key] = (lil_bi_dict[key] / total) * 100
            # filename = 'model.pkl'
            # with open(filename, 'wb') as f:
            #     pickle.dump(bi_dict, f)
            return bi_dict
        else:
            return

        #how I was originally sharing the dictionaries, need to use pickle
        # self.bi_dict = bi_dict
        # self.uni_dict = uni_dict

    
    def predict_next_word(self, inputs, deterministic):
        dictionary = {}
    
        #if unigram
        if type(inputs) == str :
            with open("PathToUnigramModel.p",'rb') as f:
                dictionary = pickle.load(f)
            #if len(dictionary.values()[0]) != 1:
             #   print("error attempting to perform bigram prediction on unigram model")
              #  return
            if inputs in dictionary:
                lil_uni_dict = dictionary[inputs]
                #returns highest probability word
                if deterministic:
                    return max(lil_uni_dict, key = lil_uni_dict.get)
                #uses random.choice to randomly selected a word, but uses weighted values
                else:
                    weights = list(lil_uni_dict.values())
                    words = list(lil_uni_dict.keys())
                    return random.choices(population = words,weights = weights, k = 1)[0]
            else:
                print("Error: Word not in vocabulary")
                return
        #if bigram
        elif len(inputs) == 2:
            #if len(dictionary.values()[0]) != 2:
             #   print("error attempting to perform unigram prediction on bigram model")
              #  return
            with open("PathToBigramModel.p",'rb') as f:
                dictionary = pickle.load(f)
            if inputs in dictionary:
                lil_bi_dict = dictionary[inputs]
                #returns highest probability word
                if deterministic:
                    return max(lil_bi_dict, key = lil_bi_dict.get)
                #uses random.choice to randomly selected a word, but uses weighted values
                else:
                    weights = list(lil_bi_dict.values())
                    words = list(lil_bi_dict.keys())
                    return random.choices(population = words,weights = weights, k = 1)[0]
            else:
                print("Error: Word not in vocabulary")
                return





class BPE:
    def __init__(self, vocabulary):
        self.vocabulary = vocabulary
    
    def train(self,data,*args):
        C = list(data)
        #C = [item.replace(" ", "</w>") for item in C]
        vocab = set(C)
        vocab = list(vocab)
        k = 500
        if args:
            k = args[0]
        
        for index in range((int)(k)):
            #pairs = C.copy()
            pairs = [C[i] + C[i+1] for i in range(len(C)-1)]
            dict_freq = {}
            loops = (int)(len(pairs))
            i = 0
            
            #use dictionary to find most common group of chars
            while i < loops:
                word = pairs[i]
                if word not in dict_freq:
                    dict_freq[word] = 1
                else:
                    dict_freq[word] += 1
                i+=1
            highest_freq = max(dict_freq, key=dict_freq.get)
            vocab.append(highest_freq)
            
            length = len(highest_freq)
            i = 0
            new_C = []
            while i < len(C)-length:
                count = 1
                word = C[i]
                while len(word) < length:
                    word += C[i+count]
                    count+=1
                if i < len(C) - length + 1 and word == highest_freq:
                    new_C.append(highest_freq)
                    i += count
                else:
                    new_C.append(C[i])
                    i += 1
            C = new_C
            
            '''
            while i < len(C)-length:
                word = "".join(C[i:i+length])
                if(word == highest_freq):
                    C[i:i+length] = [word]
                i+=1
            '''
                
                
            
            
            
            
            """loops = (int)(len(pairs)/2)
            for i in range(loops):
                pairs[i:i+2]=[pairs[i]+pairs[i+1]]
            print("loops done")
            #sets up dictionary and list for pairs
            distinct_pairs = list(set(pairs))
            frequencies = dict.fromkeys(distinct_pairs,0)

            #tracks the string value with the highest frequency
            highest_freq = pairs[0]

            #find the highest frequency pair
            for i in range(len(distinct_pairs)):
                current_pair = distinct_pairs[i]
                current_freq = pairs.count(current_pair)
                frequencies[current_pair] = current_freq
                if(current_freq>pairs.count(highest_freq)):
                    highest_freq = current_pair
            
            #replace all occurences in C 
            length = len(highest_freq)
            for i in range(len(C)-length):
                word = "".join(C[i:i+length])
                if(word == highest_freq):
                    C[i:i+length] = [word]
                
            print((str)(index) + " " + highest_freq)
            vocab.add(highest_freq)"""
        self.vocabulary = vocab
        return vocab
    
    def tokenize(self, data):
        tokens = []
        tokenIDs = []
        data_list = list(data)
        vocabulary = []
        
        with open("PathToBPEModel.p",'rb') as f:
            vocabulary = pickle.load(f)

        #get tokens that are larger than 1
        vocab = [x for x in vocabulary if len(x) > 1]
        #print(sorted(vocab))


        for token in vocab:
            #check to see if token exists by iterating through each starting position in the list
            index = 0
            while index < len(data_list):
                word = data_list[index]
                count = 1
                while(len(word) < len(token)):
                    if(index+count >= len(data_list)):
                        break
                    word += data_list[index+count]
                    count += 1
                if(word == token):
                    #merge into token
                    data_list[index:index+count] = [token]
                    #print(token)
                    index+=1
                else:
                    index += 1
                    
        
        vocab = list(vocabulary)
        #print(sorted(vocab))
        tokens = data_list.copy()

        #put the vocabulary index of the current_token into the tokenIDs
        i=0
        while i < len(tokens):
            current_token = tokens[i]
            tokenIDs.append(vocab.index(current_token))
            i+=1

        return (tokens,tokenIDs)

        

def main():
    #print("hello")
    parser = argparse.ArgumentParser()

    parser.add_argument("method", type=str)

    #moby dick reference
    parser.add_argument("--data",type=str)

    #for model being saved/loaded w pickle
    parser.add_argument("--save",type=str)
    parser.add_argument("--load",type=str)

    #tokenize only
    parser.add_argument("--text", nargs='*',type=str)

    #train_ngram only
    parser.add_argument("--n",type=int)

    #predict_ngram only
    parser.add_argument("--word", nargs='*',type=str)
    parser.add_argument("--nwords", type=int)
    parser.add_argument("--d", action="store_true")


    args = parser.parse_args()
    
    

    match args.method:
        case "train_ngram":
            if args.n:
                if args.n == 1 | args.n == 2:
                    ngram = NGram(args.n)
                    model = ngram.train(args.data)
                    filename = 'PathToUnigramModel.p'
                    if args.n == 2:
                        filename = 'PathToBigramModel.p'
                    if args.save:
                        filename = args.save
                    with open(filename, 'wb') as f:
                        pickle.dump(model, f)
            return
        case "predict_ngram":
            #need to have words to start with
            ngram = NGram(1)
            sentence = ""
            if args.word:
                sentence = args.word
            else:
                print("error: must put in words")
                return
            
            #get iteration count
            loops = 1
            if args.nwords:
                loops = args.nwords
            
            #get d
            #deterministic = False
            #if args.d:
            #    deterministic = True
                
            #predict words for unigram/bigram
            words = sentence
            sentence = words[0]
            if len(words) >1:
                sentence += " " + words[1]
            #print(args.d)
            if len(words) == 2:
                for i in range(loops):
                    word = (str)(ngram.predict_next_word((words[-2],words[-1]),args.d))
                    words.append(word)
                    sentence += " " + word
            elif len(words) == 1:
                for i in range(loops):
                    word = (ngram.predict_next_word((words[-1]),args.d))
                    if word is None:
                        break
                    word = (str)(word)
                    words.append(word)
                    sentence += " " + word
            print(sentence)
            return
        case "train_bpe":
            bpe = BPE([])
            model = bpe.train(args.data,500)
            filename = 'PathToBPEModel.pkl'
            if args.save:
                filename = args.save
            with open(filename, 'wb') as f:
                pickle.dump(model, f)
        case "tokenize":
            bpe = BPE([])
            sentence = ""
            for index in range(len(args.text)):
                sentence += args.text[index] + " "
            sentence = sentence.rstrip()
            tup = bpe.tokenize(sentence)
            print(tup)
            return
        


if __name__ == '__main__':
    main()