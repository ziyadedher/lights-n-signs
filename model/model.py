from typing import List
from numpy

'''Output object that will contain both the class and the bounding boxes for all objects
'''
class Output2D:
    def __init__(self, bb=None, classes=None):
        self.bb = bb
        self.classes = classes

    def setBB(self, bb: numpy.ndarray):
        self.bb = bb

    def setClasses(self, classes: List[str]):
        self.classes = classes

    def getClasses(self) -> List[str]:
        return self.classes

    def getBB(self) -> numpy.ndarray:
        return self.bb


'''This is the abstract class that will be inherited by every python model so that it can be standardized
'''
class Model:
    def __init__(self):
        pass

    def predict(self, image: List[List[int]]) -> Output2D:
        '''This will be implemented to automatically calculate the bounding boxes
        '''
