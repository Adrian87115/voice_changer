import generator as g
import discriminator as d
import domain_classifier as dc

class Model():
    def __init__(self):
        self.generator = g.Generator()
        self.discriminator = d.Discriminator()
        self.domain_classifier = dc.DomainClassifier()

    def train(self):
        pass
