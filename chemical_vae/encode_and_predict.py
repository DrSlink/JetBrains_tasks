from chemvae import mol_utils as mu
from chemvae.vae_utils import VAEUtils

if __name__ == "__main__":
    molecules = ['Cc1ccc(S2(=O)=NC(=O)Nc3ccccc32)cc1',
                 'CN(Cc1ccc2c(c1)C(=O)CC2)C(=O)OC(C)(C)C',
                 'COC(=O)C1CCC(Oc2ccc(NC(=O)C(=O)NN)cn2)CC1']
    vae = VAEUtils(directory='models/zinc_properties')
    with open('results.txt', 'w') as res_file:
        result = ''
        for molecule in molecules:
            smiles_1 = mu.canon_smiles(molecule)
            result += 'Was:\n'
            result += molecule + '\n\n'
            X_1 = vae.smiles_to_hot(smiles_1, canonize_smiles=True)
            z_1 = vae.encode(X_1)
            result += 'Encoded:\n'
            result += str(z_1) + '\n\n'
            X_r = vae.decode(z_1)
            result += 'After cycle of encoding-decoding:\n'
            result += str(vae.hot_to_smiles(X_r, strip=True)[0]) + '\n\n'
            result += 'Properties [qed, SAS, logP]:\n'
            y_1 = vae.predict_prop_Z(z_1)[0]
            result += str(y_1) + '\n\n\n'
        res_file.write(result)
