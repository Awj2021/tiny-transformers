env_name="tiny"
miniconda=/mnt/fast/nobackup/scratch4weeks/wa00433/miniconda3
cd /mnt/fast/nobackup/scratch4weeks/wa00433/projects/repos/tiny-transformers
# First Step for Training Local Guildance.
$miniconda/envs/$env_name/bin/python3.8 run_net.py --mode train --cfg configs/resnet/r-56_c4.yaml
# Second Step for Vit.
# $miniconda/envs/$env_name/bin/python3.8 run_net.py --mode train --cfg configs/deit/deit-ti_c100_ours.yaml

