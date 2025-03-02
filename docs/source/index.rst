================
Solverz' Utility
================

This lib provides the simulation routines for integrated energy systems, including electric power, natural gas and heat.

A simple usage of this library can be

.. code-block:: python

    from SolUtil import GasFlow
    gf = GasFlow('belgium.xlsx')
    gf.run()
    gf.Pi # get the node Pressure results


The required `.xlsx` data format can be found in SolUtil/energyflow/test directory for reference.

Contents
========

.. autoclass:: SolUtil.energyflow.gas_flow.GasFlow
   :no-members:

.. autoclass:: SolUtil.energyflow.dhs_flow.DhsFlow
   :no-members:

.. autoclass:: SolUtil.energyflow.power_flow.PowerFlow
   :no-members:
