#create a class which instantiate another class
#

class Gnappo(object):
    def __init__(self):
        self.myself = "I'm Gnappo"

class InstanceGnappo(object):
    def __init__(self, gnappo_to_instantiate):
        self.MyGnappo = gnappo_to_instantiate()


my_g = InstanceGnappo(Gnappo)