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

        self.interaction_count = 0

    def add_positive(self, images, positions):
        self.initiation.add_positive_examples(images, positions)
        self.initiation.fit_classifier()
        self.interaction_count += 1

    def add_negative(self, images, positions):
        self.initiation.add_negative_examples(images, positions)
        self.initiation.fit_classifier()
        self.interaction_count += 1

    def fit_initiation(self):
        self.initiation.fit_classifier()

    def can_initiate(self, pos):
        return self.initiation.predict(pos)

    def can_terminate(self, pos):
        pos = pos if not isinstance(pos, dict) else (pos['player_x'], pos['player_y'])
        if pos[0] <= (self.termination[0]+self.epsilon) and \
            pos[0] >= (self.termination[0]-self.epsilon) and \
            pos[1] <= (self.termination[1]+self.epsilon) and \
            pos[1] >= (self.termination[1]-self.epsilon):
            return True
        return False