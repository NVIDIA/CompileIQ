CompileIQ Documentation
=======================

.. toctree::
   :hidden:
   :maxdepth: 3
   :caption: Overview

   install
   getting_started

.. toctree::
   :hidden:
   :maxdepth: 3
   :caption: Nvidia Compiler Tuning

   compilers_overview
   booster_packs
   Tuning PTXAS <ptx_spill_example>
   Tuning NVCC <nvcc_example>
   Tuning PTXAS in Triton <triton_example>
   benchmarking

.. toctree::
   :hidden:
   :maxdepth: 3
   :caption: Advanced Usage

   workers
   normalization

.. toctree::
   :hidden:
   :maxdepth: 3
   :caption: More Examples

   xgboost
   experiment_tracking
   flashinfer_booster

.. toctree::
   :hidden:
   :maxdepth: 3
   :caption: Developer Resources

   api
   evolutionary


What is CompileIQ?
------------------

CompileIQ is a Hyper-Parameter Optimization Engine (HPO), based on evolutionary algorithms and fine-tuned to work well tuning controls of NVIDIA compilers.

This documentation portal enables you to:

* Learn about the new Advanced Controls interface of NVIDIA Compilers
* Make CompileIQ tune Advanced Controls to maximise a metric of interest
* Use CompileIQ to simultaneously adjust Advanced Controls and application parameters like Block and Batch sizes.


What can be expected from using CompileIQ?
------------------

For highly optimized workloads CompileIQ can find anywhere from 2% to 3% improvements. For sub-optimal workloads one can see upwards of 10% to 15% gains.

* Dive into the :doc:`Getting Started guide <getting_started>` to get started quickly.
* Get extra performance now with pre-made solutions from our :doc:`Booster Packs <booster_packs>`.
* Learn more about the new Compiler Controls interface by :doc:`Tuning NVIDIA Compilers <compilers_overview>`.
* Peruse our :doc:`API Documentation <api>` to get detailed information about our Python package.

