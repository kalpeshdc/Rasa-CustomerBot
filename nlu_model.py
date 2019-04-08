from rasa_nlu.model import Trainer
from rasa_nlu.training_data import load_data
from rasa_nlu import config
from rasa_nlu.model import Interpreter, Metadata

def train_model(data, configuration, model_dir):
    training_data = load_data(data)
    trainer = Trainer(config.load(configuration))
    trainer.train(training_data)
    model = trainer.persist(model_dir, fixed_model_name='customer')
    return model

def run_model(model):
    interpreter = Interpreter.load(model)
    print(interpreter.parse(u"I am planning my to order an 829 router. How much does it cost?"))


if __name__ == '__main__':
    model_directory = train_model('data/data.json', 'nlu_config.yml', 'models/nlu')
    run_model(model_directory)