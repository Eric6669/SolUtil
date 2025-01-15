from .dhs_flow import DhsFlow
from .hydraulic_mdl import hydraulic_opt_mdl


# %%
class DhsFaultFlow:

    def __init__(self,
                 df: DhsFlow,
                 fault_pipe,
                 fault_location):
        self.pipe_from = df.pipe_from
        self.pipe_to = df.pipe_to

    def relabel(self):
        pass
