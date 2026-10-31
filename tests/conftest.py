import pytest

from agents import default_agents
from data.sample_data import generate_dataset
from engine.experiment import ExperimentParams, run_experiment


@pytest.fixture(scope="session")
def dataset():
    return generate_dataset(n_stocks=120, n_periods=12, seed=7)


@pytest.fixture(scope="session")
def agents():
    return default_agents()


@pytest.fixture(scope="session")
def experiment(dataset):
    params = ExperimentParams(seed=7, n_stocks=120, n_periods=12, top_n=10, random_trials=50)
    return run_experiment(params, dataset=dataset)


@pytest.fixture(scope="session")
def result(experiment):
    return experiment.backtest


@pytest.fixture(scope="session")
def benchmarks(experiment):
    return experiment.benchmarks
