
import subprocess

# subprocess.run(f'mv /home/huke/project/GNNome-assembly/data/references_Maize /data2/huke', shell=True)

# for i in range(1, 23):
# #     # subprocess.run(f'mkdir chr{i}',shell=True)
#     # subprocess.run(f'mv /data2/huke/real/chr{}')
#     subprocess.run(f'mv /home/huke/project/GNNome-assembly/data/experiments/SymGated/test_{i}_SymGated /data/huke/SymGated', shell=True)

# for i in range(4, 23):
    # subprocess.run(f'mkdir /data2/huke/mummerplot/{i}_test_PathNN', shell=True)

# delta-filter -i 90 -l 10000 -q out.delta  > 13.delta.filter

# mummerplot --png 13.delta.filter

# show-coords -r -T 13.delta.filter > 13.coords

for i in range(1, 23):
    subprocess.run(f'minimap2 /home/huke/project/GNNome-assembly/data/references/chromosomes/chr{i}.fasta /data2/huke/experiments/Sage_real/test_{i}_Sage_real/assembly/0_assembly.fasta > {i}.paf', shell=True)