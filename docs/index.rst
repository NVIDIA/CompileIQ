CompileIQ Documentation
======================================

.. toctree::
   :hidden:
   :maxdepth: 3
   :caption: Overview

   install
   getting_started

.. toctree::
   :hidden:
   :maxdepth: 3
   :caption: NVIDIA Compiler Tuning

   compilers_overview
   search_space_release_testing
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
   Search Strategy <search_strategy>


What is CompileIQ?
------------------------------------------

CompileIQ is a hyperparameter optimization engine (HPO) designed for tuning NVIDIA compiler controls.

This documentation portal enables you to:

* Learn about the new Advanced Controls interface of NVIDIA Compilers.
* Make CompileIQ tune the compiler's Advanced Controls to maximise a metric of interest.
* Use CompileIQ to simultaneously adjust Advanced Controls and application parameters like block and batch sizes.


What can be expected from using CompileIQ?
------------------------------------------

For highly optimized workloads, CompileIQ has shown 2% to 3% improvements in some cases. Less optimized workloads may see larger gains, but actual results depend on the workload, hardware, metric, and available optimization headroom.

* Dive into the :doc:`Getting Started guide <getting_started>` to get started quickly.
* Get extra performance now with pre-made solutions from our :doc:`Booster Packs <booster_packs>`.
* Learn more about the new Compiler Controls interface by :doc:`Tuning NVIDIA Compilers <compilers_overview>`.
* Peruse our :doc:`API Documentation <api>` to get detailed information about our Python package.
