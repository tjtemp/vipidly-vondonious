## TODO: currently not working

import matplotlib.pyplot as plt, mpld3
import seaborn as sns
import pandas as pd
import numpy as np
N=10
data = pd.DataFrame({"x": np.random.randn(N),
                     "y": np.random.randn(N),
                     "size": np.random.randint(20,200, size=N),
                     "label": np.arange(N)
                     })


scatter_sns = sns.lmplot("x", "y",
           scatter_kws={"s": data["size"]},
           robust=False, # slow if true
           data=data, size=8)
fig = plt.gcf()

ax = plt.gca()
pts = ax.get_children()[3]
tooltip = mpld3.plugins.PointLabelTooltip(pts, labels=list(data.label))


mpld3.plugins.connect(fig, tooltip)

mpld3.display(fig)

