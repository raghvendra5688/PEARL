# source me: sets up effichem env without needing shell hooks
export PY=/export/home/rmall/micromamba/envs/effichem/bin/python
export LD_LIBRARY_PATH=/export/home/rmall/micromamba/envs/effichem/lib:$LD_LIBRARY_PATH
export OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
