from .Step1 import Step1
from .Step2 import Step2



def get_dataset(name):
    return {
        "step1": Step1,
        "step2": Step2,
    }[name]
