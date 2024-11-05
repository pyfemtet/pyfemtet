from tqdm import tqdm

from hc.problem.spiral import Spiral
from hc.problem.hyper_sphere import HyperSphere
from hc.problem.spots_in_square import SpotsInSquare
from hc.sampler.bo_pof_class import BayesSampler
from hc.sampler.random import RandomSampler

import hc.sampler.bo_pof_class as bo_pof_class


if __name__ == '__main__':

    bo_pof_class.threshold = 0.5
    bo_pof_class.gamma = 1.

    # 問題の設定
    # problem = Spiral()
    problem = SpotsInSquare()
    # problem = HyperSphere(2)

    # 最適化手法の設定
    random_manager = RandomSampler(problem)
    bayesian_manager = BayesSampler(problem)

    # 拘束違反の際に目的関数が悪い値を返したことにする設定
    bayesian_manager.return_wrong_if_infeasible = False
    bayesian_manager.wrong_rate = 1.  # 1: 最悪, 0: 最良, >1: 最悪よりもより悪い

    # サロゲートモデルを作成する為の initial sampling
    for i in tqdm(range(100), 'random sampler'):
        random_manager.sampling(False)

    # 一旦結果を表示しながらサンプリング
    random_manager.sampling(True)

    # 別の手法で続きから実施する場合は
    # まずデータフレームを引き継ぐ
    bayesian_manager.df = random_manager.df

    # サロゲートモデルを更新しながらベイズ最適化
    for i in tqdm(range(5), 'bayesian pof sampler'):
        bayesian_manager.sampling(True)

    # 記録ファイル名を作成
    method_name = 'bo'
    if bo_pof_class.gamma > 0:
        gamma = bo_pof_class.gamma
        method_name += f'_pof {gamma=}'
    if bayesian_manager.return_wrong_if_infeasible:
        wrong_rate = bayesian_manager.wrong_rate
        method_name += f' {wrong_rate=}'

    # 記録
    bayesian_manager.sampling(
        True,
        csv_path=f'{method_name}.csv',
        figure_path=f'{method_name}.png',
    )
