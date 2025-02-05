print('a')

class Anotator:
    def  __init__(self, stoi = None, lookup = None):
        self.texts = []
        self.emotions = []
        self.stoi = None
        self.lookup = None
    def receive_input(self,text):
        i = 1
        self.clear()
        user_input = input(f"Write utterance {i}: ")
        while user_input != '':
            i += 1
            self.texts.append(user_input)
            user_input = input(f"Write utterance {i}: ")
        print(f"For each utterance, select an emotion in: {self.lookup.keys()}")
        for j in self.texts:
            emotion_input = input(f'Select and emotion for {j}\nEmotion: ')
            while emotion_input not in self.lookup.keys():
                emotion_input = input(f'Select and emotion for: {j} \nEmotion: ')
            self.emotions.append(emotion_input)
    
    def clear(self):
        self.texts = []
        self.emotions = []

    def load_lookup(self, path):
        pass
    def load_stoi(self, path):
        pass
    def preprocess(self):
        pass
