import os
from math import pi

import matplotlib.pyplot as plt
import numpy as np
from Bio.PDB import PDBList, PDBParser, PPBuilder

if __name__ == "__main__":
    result_folder = 'results'

    if not os.path.exists(result_folder):
        os.makedirs(result_folder)

    for PDB_id in ['1MBN', '5KI0', '4GCR']:
        pdb_path = PDBList().retrieve_pdb_file(pdb_code=PDB_id,
                                               pdir='PDB',
                                               file_format='pdb')

        figure, axes = plt.subplots(figsize=(5.5, 5), dpi=100)
        axes.set_title(PDB_id)

        phi_psi_dict = {}
        for model in PDBParser().get_structure(id=None, file=pdb_path):
            for chain in model:
                peptides = PPBuilder().build_peptides(chain)
                for peptide in peptides:
                    phi_psi = peptide.get_phi_psi_list()
                    for aminoacid, angles in zip(peptide, phi_psi):
                        residue = aminoacid.resname + str(aminoacid.id[1])
                        phi_psi_dict[residue] = angles
        x, y = [], []
        for key, value in phi_psi_dict.items():
            if value[0] and value[1]:
                x.append(value[0] * 180 / pi)
                y.append(value[1] * 180 / pi)

        axes.set_aspect('equal')
        axes.set_xlabel('\u03C6')
        axes.set_ylabel('\u03C8')
        axes.set_xlim(-180, 180)
        axes.set_ylim(-180, 180)
        axes.set_xticks([-180, -135, -90, -45, 0, 45, 90, 135, 180], minor=False)
        axes.set_yticks([-180, -135, -90, -45, 0, 45, 90, 135, 180], minor=False)
        plt.axhline(y=0, color='k', lw=0.5)
        plt.axvline(x=0, color='k', lw=0.5)
        plt.grid(b=None, which='major', axis='both', color='k', alpha=0.2)

        Z = np.fromfile('density_estimate.dat')
        Z = np.reshape(Z, np.mgrid[min(x):max(x):100j, min(y):max(y):100j][0].shape)
        data = np.log(np.rot90(Z))
        axes.imshow(data, cmap=plt.get_cmap('viridis'),
                    extent=[-180, 180, -180, 180],
                    alpha=0.75)

        data = np.rot90(np.fliplr(Z))
        axes.contour(data, colors='k', linewidths=0.5,
                     levels=[10 ** i for i in range(-7, 0)],
                     antialiased=True, extent=[-180, 180, -180, 180], alpha=0.65)
        plt.scatter(x, y, marker='.', s=3, c='k')
        plt.savefig(os.path.join(result_folder, PDB_id + ".png"))
