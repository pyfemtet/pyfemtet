from tqdm import tqdm

from hc.problem.spiral import Spiral
from hc.problem.hyper_sphere import HyperSphere
from hc.problem.spots_in_square import SpotsInSquare
from hc.sampler.bo import BayesSampler
from hc.sampler.random import RandomSampler


if __name__ == '__main__':
    problem = HyperSphere(6)
    problem.r_upper = 1e10
    random_manager = RandomSampler(problem)
    bayesian_manager = BayesSampler(problem)

    for i in tqdm(range(100), 'random sampler'):
        random_manager.sampling()

    # 出力する場合は True (表示)、パス（画像）を渡す
    random_manager.show_figure()
    random_manager.save_figure('test_image.png')
    random_manager.save_figure('test_image.svg')

    # 別の手法で続きから実施する場合は
    # まずデータフレームを引き継ぐ
    bayesian_manager.df = random_manager.df

    for i in tqdm(range(5), 'bayesian sampler'):
        bayesian_manager.sampling()
        bayesian_manager.show_figure()
