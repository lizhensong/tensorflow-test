from gensim.models import FastText
sentences_1 = [["cat", "say", "meow"], ["dog", "say", "woof"]]
sentences_2 = [["dude", "say", "wazzup!"]]

model = FastText(min_count=1)
model.build_vocab(sentences_1)
model.train(sentences_1, total_examples=model.corpus_count, epochs=model.epochs)

model.build_vocab(sentences_2, update=True)
model.train(sentences_2, total_examples=model.corpus_count, epochs=model.epochs)
