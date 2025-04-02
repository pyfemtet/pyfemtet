import os
import numpy as np
import pandas as pd
from glob import glob
import plotly.graph_objects as go

obj_name = 'distance'

new_df = None

for path in glob('*.csv'):
    filename = os.path.basename(path).removesuffix('.csv')
    print(filename)

    ret = tuple(map(lambda string: string.split('_')[0], filename.split('=')[1:]))
    print(ret)

    o_noise, f_noise, seed = ret
    print(o_noise, f_noise, seed)

    df = pd.read_csv(path, header=2)
    print(df.head())

    obj_values = df[obj_name].dropna().values
    print(obj_values)

    minimum_trace = [min(obj_values[:int(i+1)]) for i, _ in enumerate(obj_values)]
    print(minimum_trace)

    pdf = pd.DataFrame(dict(
        succeeded_trial=range(len(minimum_trace)),
        direction=minimum_trace,
        observation_noise=[o_noise] * len(minimum_trace),
        feasibility_noise=[f_noise] * len(minimum_trace),
        seed=[seed] * len(minimum_trace),
    ))
    print(pdf)

    if new_df is None:
        new_df = pdf
    else:
        new_df = pd.concat([new_df, pdf])

    print(len(new_df))


# 条件でグルーピング
gdf = new_df.groupby(['observation_noise', 'feasibility_noise', 'succeeded_trial'], as_index=False)
mean_df = gdf.mean(numeric_only=True)
std_df = gdf.std(numeric_only=True)


fig = go.Figure()

colors = ['red', 'green', 'blue', 'orange']
for o_noise in np.unique(mean_df['observation_noise']):
    for f_noise in np.unique(mean_df['feasibility_noise']):
        idx = (mean_df['observation_noise'] == o_noise) * (mean_df['feasibility_noise'] == f_noise)
        x = mean_df[idx]['succeeded_trial'].values
        y = mean_df[idx]['direction'].values
        s = std_df[idx]['direction'].values

        color = colors.pop(0)

        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                name=f'{o_noise=}, {f_noise=}',
                mode='lines',
                line=dict(color=color),
                legendgroup=f'{o_noise=}, {f_noise=}',
            )
        )

        fig.add_trace(
            go.Scatter(
                x=list(x) + list(x)[::-1],
                y=list(y - s) + list(y + s)[::-1],
                name=f'{o_noise=}, {f_noise=} (std)',
                fillcolor=color,
                opacity=0.1,
                showlegend=False,
                fill='toself',
                legendgroup=f'{o_noise=}, {f_noise=}',
            )
        )
# fig.update_layout(
#     yaxis=dict(title=dict(text='Wind speed (m/s)')),
#     title=dict(text='Continuous, variable value error bars'),
#     hovermode="x"
# )
fig.show()
