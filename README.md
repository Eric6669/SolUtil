# Solverz Utilities

This lib provides the simulation routines for integrated energy systems, including electric power, natural gas and heat.

A simple usage of this library can be

```python
from SolUtil import GasFlow
gf = GasFlow('belgium.xlsx')
gf.run()
print(gf.Pi) # get the node Pressure results
print(gf.f) # get the node pipe mass-flow results
```

The required `.xlsx` data format can be found in SolUtil/energyflow/test directory for reference.
