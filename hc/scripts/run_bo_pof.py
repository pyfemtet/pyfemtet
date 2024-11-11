from tqdm import tqdm

from hc.problem.spiral import Spiral
from hc.problem.hyper_sphere import HyperSphere
from hc.problem.spots_in_square import SpotsInSquare, HyperSpotsInSquare
from hc.sampler.bo_pof import BayesSampler
from hc.sampler.random import RandomSampler

import hc.sampler.bo_pof as bo_pof


if __name__ == '__main__':

    bo_pof.threshold = 0.5
    bo_pof.gamma = 1.

    # 問題の設定
    # problem = Spiral()
    # problem = HyperSphere(2)
    # problem = SpotsInSquare()
    dim = 9
    radii = {
        2: 0.05,
        6: 0.4,
        9: 0.95,
    }[dim]
    spots = [dict(center=[0.]*dim, radii=radii)]
    for i in range(dim):
        center = [0. for _ in range(dim)]
        center[i] = 1. - radii
        spots.append(dict(center=center, radii=radii))
    problem = HyperSpotsInSquare(spots, dim)

    # 最適化手法の設定
    random_manager = RandomSampler(problem)
    bayesian_manager = BayesSampler(problem)

    # 拘束違反の際に目的関数が悪い値を返したことにする設定
    bayesian_manager.return_wrong_if_infeasible = False
    bayesian_manager.wrong_rate = 1.  # 1: 最悪, 0: 最良, >1: 最悪よりもより悪い

    # 最初は feasible region で実行する
    random_manager.sampling(problem.initial_points[0])

    # サロゲートモデルを作成する為の initial sampling
    # for i in tqdm(range(5), 'random sampler'):
    #     random_manager.sampling()
    while len(random_manager.x_feasible) < 5:
        random_manager.sampling()
        print(
            f'{len(random_manager.x)} sampling '
            f'/ {len(random_manager.x_feasible)} feasible '
            f'/ (hit rate: '
            f'{(float(len(random_manager.x_feasible))/float(len(random_manager.x)))*100:.0f}%'
            f')'
        )

    # 一旦結果を表示
    random_manager.show_figure()

    # 別の手法で続きから実施する場合は
    # まずデータフレームを引き継ぐ
    bayesian_manager.df = random_manager.df

    # サロゲートモデルを更新しながらベイズ最適化
    for i in tqdm(range(5), 'bayesian pof sampler'):
        bayesian_manager.sampling()
        bayesian_manager.show_figure()

    # 記録ファイル名を作成
    method_name = 'bo'
    if bo_pof.gamma > 0:
        gamma = bo_pof.gamma
        method_name += f'_pof {gamma=}'
    if bayesian_manager.return_wrong_if_infeasible:
        wrong_rate = bayesian_manager.wrong_rate
        method_name += f' {wrong_rate=}'

    # 記録
    bayesian_manager.save_csv(method_name + '.csv')
    # bayesian_manager.save_figure(method_name + '.png')
    # bayesian_manager.save_figure(method_name + '.svg')
