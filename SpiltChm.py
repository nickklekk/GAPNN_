import gzip
import os
from Bio import SeqIO
import subprocess

for i in range(1, 11):
    pth = '/home/huke/project/GNNome-assembly/data/simulated_Maize'
    subprocess.run(f'mkdir chr{i}', shell=True, cwd='/home/huke/project/GNNome-assembly/data/simulated_Maize')
    subprocess.run(f'mkdir raw processed info raven_output graphia', shell=True, cwd =os.path.join(pth, f'chr{i}') )

# # chm_path = os.path.join(ref_path, 'CHM13')
# chr_path = os.path.join('/home/huke/project/GNNome-assembly/data/references_Maize/chromosomes')

# # 棉花ref
# # chm13_path = os.path.join('/home/huke/project/GNNome-assembly/data/references_CRI/CRI/genome.Gaus.CRI.fa.gz')

# # 茶树ref 
# # chm13_path = os.path.join('/home/huke/project/GNNome-assembly/data/references_Tea/CHM_ref/ref.fa')

# # 番茄ref
# chm13_path = os.path.join('/home/huke/project/GNNome-assembly/data/references_Tomato/CHM_ref/GWHBAUD00000000.genome.fasta.gz')

# #
# chm13_path = os.path.join('/home/huke/project/GNNome-assembly/data/Zm-Mo17-REFERENCE-CAU-2.0.fa.gz')

# if len(os.listdir(chr_path)) == 0:
#         # Parse the CHM13 into individual chromosomes
#         print(f'SETUP::download:: Split per chromosome')
#         with gzip.open(chm13_path, 'rt') as f:
#         # with open(chm13_path, 'rt') as f:
#             for record in SeqIO.parse(f, 'fasta'):
#                 SeqIO.write(record, os.path.join(chr_path, f'{record.id}.fasta'), 'fasta')