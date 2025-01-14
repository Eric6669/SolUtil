import numpy as np
import pandas as pd
from ..dhs_flow import DhsFlow


def test_dhs_flow(datadir):

    bench = pd.read_excel(datadir / 'BarryIsland.xlsx',
                          sheet_name=None,
                          engine='openpyxl',
                          index_col=None
                          )

    df = DhsFlow(datadir / 'BarryIsland.xlsx')
    df.phi *= np.asarray(bench['load_fac']['load_fac'])
    df.run(tee=False)

    for varname in ['m', 'Ts', 'Tr']:
        np.testing.assert_allclose(df.__dict__[varname],
                                   np.asarray(bench[f'bench_{varname}'][varname]),
                                   rtol=1e-4,
                                   atol=1e-5)

    np.testing.assert_allclose(df.phi[df.slack_node], np.asarray(bench['bench_phi_slack']['phi_slack']))
