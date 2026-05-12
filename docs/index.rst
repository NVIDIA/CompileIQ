CompileIQ Documentation
=======================

.. toctree::
   :hidden:
   :maxdepth: 3
   :caption: Overview

   primer
   install
   getting_started

.. toctree::
   :hidden:
   :maxdepth: 3
   :caption: More Examples

   xgboost
   experiment_tracking

.. toctree::
   :hidden:
   :maxdepth: 3
   :caption: Nvidia Compiler Tuning

   compilers_overview
   nvcc_example
   ptx_spill_example
   triton_example
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
   :caption: Programmer's Guide

   api



What is CompileIQ?
------------------

CompileIQ is an evolutionary-based HPO. The internal parameters were fine-tuned to work well with Nvidia's compilers.

CompileIQ's primary use is evolutionary optimization over a multidimensional problem space. More usefully, CompileIQ's goal is to target performance improvements in any application capable of deterministically altering its behavior according to a set of configuration settings or a defined search space.
