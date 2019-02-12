from nltk.corpus import wordnet

def get_synset_from_label(label):
    return wordnet.synset_from_pos_and_offset(label[0], int(label[1:]))

def get_label_from_synset(synset):
    return synset.pos() + str(synset.offset()).zfill(8)

def get_hyponyms(label):
    return [get_label_from_synset(synset) for synset in get_synset_from_label(label).hyponyms()]

def get_all_hyponyms(label):
    hyponyms = lambda s: s.hyponyms()
    return [get_label_from_synset(synset) for synset in get_synset_from_label(label).closure(hyponyms)]

def get_hypernyms(label):
    return [get_label_from_synset(synset) for synset in get_synset_from_label(label).hypernyms()]

def get_all_hypernyms(label):
    hypernyms = lambda s: s.hypernyms()
    return [get_label_from_synset(synset) for synset in get_synset_from_label(label).closure(hypernyms)]

def get_label_from_word(word, valid_labels=None):
    labels = [get_label_from_synset(synset) for synset in wordnet.synsets(word)]
    if not valid_labels == None:
        labels = [label for label in labels if label in valid_labels]
    return labels[0]

def get_word_from_label(label):
    return get_synset_from_label(label).lemmas()[0].name()
