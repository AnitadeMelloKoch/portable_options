from portable.option.sets.models import PositionClassifier

class PositionSetPair():
    def __init__(self,
                images,
                positions,
                termination,
                epsilon=2):
        
        self.initiation = PositionClassifier()
        
        self.initiation.add_positive_examples(images, positions)

        self.initiation.fit_classifier()
        self.epsilon = epsilon

        termination = termination[0] if isinstance(termination, list) else termination
        self.termination = (termination['player_x'], termination['player_y'])

    def add_positive(self, images, positions):
        self.initiation.add_positive_examples(images, positions)
        self.initiation.fit_classifier()

    def add_negative(self, images, positions):
        self.initiation.add_negative_examples(images, positions)
        self.initiation.fit_classifier()

    def fit_initiation(self):
        self.initiation.fit_classifier()

    def check_initiation(self, pos):
        return self.initiation.predict(pos)

    def check_termination(self, pos):
        pos = pos if not isinstance(pos, dict) else (pos['player_x'], pos['player_y'])
        if pos[0] <= (self.termination[0]+self.epsilon) and \
            pos[0] >= (self.termination[0]-self.epsilon) and \
            pos[1] <= (self.termination[1]+self.epsilon) and \
            pos[1] >= (self.termination[1]-self.epsilon):
            return True
        return False