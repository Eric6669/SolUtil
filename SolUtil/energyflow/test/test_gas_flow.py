import numpy as np
import pandas as pd
from ..gas_flow import GasFlow


def test_gas_flow(datadir):
    gf = GasFlow(datadir/'belgium.xlsx')
    gf.run()

    df = pd.read_excel(datadir/'belgium.xlsx',
                       sheet_name=None,
                       engine='openpyxl',
                       index_col=None
                       )

    f_bench = np.asarray(df['bench_f']['f'])
    pi_bench = np.asarray(df['bench_pi']['pi'])

    np.testing.assert_allclose(gf.Pi, pi_bench)
    np.testing.assert_allclose(gf.f, f_bench)
