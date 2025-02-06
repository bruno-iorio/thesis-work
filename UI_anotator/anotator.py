from nltk.tokenize import TweetTokenizer
class Anotator:
    def  __init__(self, stoi = None, lookup = None):
        self.texts = []
        self.emotions = []
        self.stoi = stoi
        self.lookup = lookup
        self.tok = TweetTokenizer()
        self.emotion_set = set()
    def receive_input(self):
        i = 1
        self.clear()
        user_input = input(f"Write utterance {i}: ")

        for emotion in self.lookup:
            self.emotion_set.add(emotion.split('_')[0])
        while user_input != '':
            i += 1
            self.texts.append(user_input)
            user_input = input(f"Write utterance {i}: ")
        print(f"For each utterance, select an emotion in: {self.lookup.keys()}")
        for j in self.texts:
            while True:
                emotion_input = input(f'Select and emotion for: {j} \nEmotion: ')
                if emotion_input not in self.emotion_set:
                    continue
                else:
                    break
            self.emotions.append(emotion_input)
    
    def clear(self):
        self.texts = []
        self.emotions = []
    
    def preprocess(self):
        new_texts = []
        new_emotions = []
        for i in range(len(self.texts)): 
            text = self.tok.tokenize(self.texts[i])
            new_texts.extend([self.stoi[j] if j in self.stoi else self.stoi['<unk>'] for j in text])
            new_emotions.extend([self.lookup[f"{self.emotions[i]}_{j}"] if f'{self.emotions[i]}_{j}' in self.lookup.keys() else self.lookup['NE'] for j in range(len(text))])
        self.texts = new_texts
        self.emotions = new_emotions

    def load_lookup(self, path): ## Load from json
        with open(path, 'r') as file: 
            self.lookup = file.dumps(path)

    def load_stoi(self, path): ## Load from json
        with open(path, 'r') as file:
            self.stoi = file.dumps(path)

def test1():
    a = Anotator()
    a.stoi = {'<pad>':0,'<sep>':1,'<unk>':2,'oi':3,'tudo':4,'bem':5,'tchau':6}
    a.lookup = {'NE':0,'happy_0':1,'happy_1':2,'happy_2':3,'sad_0':4,'sad_1':5,'sad_2':6,'sad_3':7}
    a.receive_input()

    print(a.emotions)
    print(a.texts)

    a.preprocess()
    print(a.emotions,a.texts)
#test1()
