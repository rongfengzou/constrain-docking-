from rdkit import Chem
from rdkit.Chem import AllChem
import argparse
import os
import subprocess

def pdbqt2pdb(pdbqt, pdb):
    cmd = "obabel {} -O {}".format(pdbqt, pdb)
    subprocess.call(cmd, shell=True)

def assign_bond_order(template, docked_pose):
    """
    Assign bond order to the docked pose using the template molecule
    """
    template = Chem.SDMolSupplier(template, removeHs=True)[0]
    pdbqt2pdb(docked_pose, 'docked_pose.pdb')
    docked_pose_mol = Chem.MolFromPDBFile('docked_pose.pdb', removeHs=False)
    newMol = AllChem.AssignBondOrdersFromTemplate(template, docked_pose_mol)
    return newMol


parser = argparse.ArgumentParser(description='Assign bond order to the docked pose using the template molecule')
parser.add_argument('-t', '--template', help='Template molecule', required=True)
parser.add_argument('-d', '--docked_pose', help='Docked pose, pdbqt file', required=True)
parser.add_argument('-o', '--output', help='Output file', required=True)
args = parser.parse_args()

newMol = assign_bond_order(args.template, args.docked_pose)
newMol_H = Chem.AddHs(newMol, addCoords=True)
writer = Chem.SDWriter(args.output)
writer.write(newMol_H)
writer.close()

# remove docked_pose.pdb
os.remove('docked_pose.pdb')



