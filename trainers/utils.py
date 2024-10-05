from . import *

def get_trainer(model_name):
    if model_name == "VAE":
        trainer_class = VAETrainer
    elif model_name == "TensorContracted":
        trainer_class = TensorContractedTrainer
    elif model_name == "TensorConFormer":
        trainer_class = TensorConFormerTrainer
    elif model_name == "Transformed":
        trainer_class = TransformedTrainer
    elif model_name == "TensorConFormerEnc":
        trainer_class = TensorConFormerEncTrainer
    elif model_name == "TensorConFormerDec":
        trainer_class = TensorConFormerDecTrainer
    else:
        raise ValueError(f"Model: {model_name} not found.")
    return trainer_class
