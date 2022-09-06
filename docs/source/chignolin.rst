.. _chignolin:

Fold the mini-protein Chignolin
===============================

.. image:: ../../examples/chignolin/data/2RVD.png
  :width: 200
  :alt: folded structure of the chignolin protein	  
	   
Chignolin is an artificial mini-protein consisting of only 10 amino acid residues.
Here we use potential contrasting to learn a coarse-grained force field that can
capture the protein's folding and unfolding 

1. Download the all-atom simulation trajectory
----------------------------------------------

An all-atom simulation of chignolin is used as training data.
You can follow the link `pccg-chignolin <https://www.kaggle.com/datasets/negelis/pccg-chignolin>`_
to download the all-atom simulation trajectory.
Then you should uncompress the downloaded data and move them to the directory ``WORK_DIR/data``.
Here ``WORK_DIR`` can be any directory and wil be used as the working directory for running commands
in the tutorial.
In other words, you should change your working directory to ``WORK_DIR`` before running following commands.
In the direcotry ``WORK_DIR/data``, there should be three files:

* ``cln025_all_atom.dcd``: an all-atom simulation trajectory
* ``cln025.prmtop``: the topology file of chignolin
* ``cln025_reference.pdb``: a folded conformation of chignolin

The all-atom simulation was conducted with implicit solvent and near the folding temperature,
so the all-atom simulation trajectory contains both folded and unfolded conformations.
You can visualize the trajectory by loading it together with the topology file into softwares such as ``VMD``.

2. Convert the all-atom trajectory into a coarse-grained one
------------------------------------------------------------

