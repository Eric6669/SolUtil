from SolUtil import DhsFlow, DhsFaultFlow
import pytest

FILES = [
    'case_heat_test.xlsx',
    'case_heat_test0.xlsx',
    'case_heat_test1.xlsx'
]


# 使用 pytest 的 fixture 来加载 Excel 文件
@pytest.fixture(params=FILES)
def test_file_path(request, datadir):
    file_path = datadir / request.param
    return file_path


def test_dhs_fault_flow0(test_file_path):
    assert test_file_path.exists(), f"File {test_file_path} does not exist"

    df = DhsFlow(test_file_path)
    df.run()

    dff = DhsFaultFlow(df, 0, 0.5, fault_sys='r', dH=1)
    dff.run()

    assert dff.verify_results() < 1e-5
