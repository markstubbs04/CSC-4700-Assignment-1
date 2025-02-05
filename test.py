import pickle
from Stubbs_hnrs3035_cshw1 import NGram, BPE

ngram = NGram(2)
model = ngram.train(ngram.moby_dick)
with open("model.pkl", 'wb') as f:
    pickle.dump(model, f)
word = (str)(ngram.predict_next_word(("!","A"),False))


#test for building a sentence with unigram
ngram = NGram(1)
model = ngram.train(ngram.moby_dick)
with open("PathToUnigramModel.p", 'wb') as f:
    pickle.dump(model, f)
sentence = "whale"
for i in range(50):
    words = sentence.split()
    word = (str)(ngram.predict_next_word((words[-1]),False))
    sentence += " " + word
print(sentence)

#test for building a sentence with bigram
ngram = NGram(2)
model = ngram.train(ngram.moby_dick)
with open("PathToBigramModel.p", 'wb') as f:
    pickle.dump(model, f)
inputs = ("the","harpooneer")
sentence = inputs[0] +" "+ inputs[1]
words = sentence.split()
for i in range(50):
    word = (str)(ngram.predict_next_word((words[-2],words[-1]),False))
    words.append(word)
    sentence += " " + word
print(sentence)



#bpe train
moby_dick = ""
with open("moby_dick.txt", "r",errors='ignore') as f:
    moby_dick = f.read()
bpe = BPE([])
model = bpe.train(moby_dick,3000)
filename = 'PathToBPEModel.p'
with open(filename, 'wb') as f:
    pickle.dump(model, f)
    
    
#tokenize'
tup = bpe.tokenize("The bright red tomato was eaten by the whale!")
print("".join(tup[0]))
print(tup)
