
cd $PBS_O_WORKDIR

source ~/apps/scripts/source_conda.sh

conda activate westpa-2022.02

python westpa_frame_flux.py
