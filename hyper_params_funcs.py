import pickle
import sklearn as sk


def get_model_name(model) -> str:
    """
    recieves an sk model and outputs its name
    """
    model_name: str = model.__repr__().split('(')[0]
    if model_name.endswith('Wrapper'):
        model_name = model.model.__repr__().split('(')[0]

    return model_name


def model_problem_name(model: sk.base.BaseEstimator, problem: str) -> str:
    """
    recieves an sk model and a problem name and outputs the problem name and model that tries to solve this problem
    """
    model_name: str = get_model_name(model)

    return f'{problem}_{model_name}.pickle'


def save_model_problem_hyper_params(model: sk.base.BaseEstimator, problem: str):
    """
    recieves a model and a problem and saves the model hyper_parameters in the proper file
    """
    with open(model_problem_name(model, problem), 'wb') as handle:
        saved_params = {key: value for key, value in model.get_params().items() if key != 'model'}
        pickle.dump(saved_params, handle)


def save_problem_hyper_params(models, problem: str):
    """
    recieves a list of model and a problem and saves the models hyper_parameters in the proper file
    """

    for model in models:
        save_model_problem_hyper_params(model, problem)


def load_model_problem_hyper_params(model, problem: str, verbose=False):
    """
    recieves a model and a problem and loads the models hyper parameters from the file to witch they were last saved
    """
    with open(model_problem_name(model, problem), 'rb') as handle:
        params = pickle.load(handle)

    if verbose:
        print(f'For the problem {problem} the best hyper parameters for the estimator {get_model_name(model)} are:')
        print(get_normal_params(params))

    model.set_params(**params)


def get_normal_params(params: dict) -> dict:
    return {(key.split('model__')[1] if key.startswith('model') else key): value for key, value in params.items() if key != 'model'}


def copy_model(model):
    """
    gets a model and outputs a copy of it
    """
    return type(model)(**model.get_params())


def load_problem_hyper_params(models, problem: str, wrapper=None, verbose=False):
    """
    loads the hyper parameters of a list of models on some problem to the wrapped models if needed
    """
    used_models = [copy_model(model) for model in models]
    if wrapper is not None:
        used_models = [wrapper(model) for model in used_models]

    for model in used_models:
        load_model_problem_hyper_params(model, problem, verbose)

    return used_models


def print_best_hyper_params(models, problem: str):
    """
    prints the best hyper_params of a a list of models fo a given problem
    """
    print(f'The best hyper-parameters for the {problem} problem are:')
    print('')
    for model in models:
        print(model)
        print('')
    print('')


def print_best_model(model, problem: str):
    """
    print the best model to solve the given problem
    """
    print(f'The best model for the {problem} problem is:')
    print(model)
    print('')
