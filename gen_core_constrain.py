from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import rdFMCS
import sys
from math import isnan, isinf
from rdkit.Chem import ChemicalForceFields
from rdkit.Chem import rdMolAlign


def get_atom_mapping(mol, core):
    mcs = rdFMCS.FindMCS([mol, core], ringMatchesRingOnly=True, completeRingsOnly=True, matchChiralTag=True, timeout=100)
    mcs_mol = Chem.MolFromSmarts(mcs.smartsString)
    mol_match = mol.GetSubstructMatch(mcs_mol)
    core_match = core.GetSubstructMatch(mcs_mol)
    atom_map = {}
    for i in range(len(mol_match)):
        atom_map[mol_match[i]] = core_match[i]
    return atom_map

def constrain_embed(mol, core, atom_map):
    rms = rdMolAlign.AlignMol(mol , core, atomMap=list(atom_map.items()))
    temp_conf = Chem.Conformer()
    for mol_idx in range(mol.GetNumAtoms()):
        if mol_idx not in atom_map.keys():
            temp_conf.SetAtomPosition(mol_idx, mol.GetConformer().GetAtomPosition(mol_idx))
        else:
            temp_conf.SetAtomPosition(mol_idx, core.GetConformer().GetAtomPosition(atom_map[mol_idx]))
    
    mol.RemoveAllConformers()
    mol.AddConformer(temp_conf)
    
    ff_p = ChemicalForceFields.MMFFGetMoleculeProperties(mol, 'MMFF94s')
    ff = ChemicalForceFields.MMFFGetMoleculeForceField(mol, ff_p, confId=0)
    # add restraints
    for key in atom_map.keys():
        core_atom_position = core.GetConformer().GetAtomPosition(atom_map[key])
        virtual_site_atom = ff.AddExtraPoint(core_atom_position.x, core_atom_position.y, core_atom_position.z, fixed=True) - 1
        ff.AddDistanceConstraint(virtual_site_atom, key, 0, 0, 10.0)
    
    # restrained minimization
    ff.Initialize()
    max_minimize_iteration = 5
    for _ in range(max_minimize_iteration):
        minimize_seed = ff.Minimize(energyTol=1e-5, forceTol=1e-3)
        if minimize_seed == 0:
            break
    return mol

def remove_rot_bond(mol, bond_atoms, scaffold_atoms):
    remove_index = []
    for i in range(len(bond_atoms)):
        for j in scaffold_atoms:
            if j in bond_atoms[i]:                
                neighbor_atoms = []
                rot_a1 = mol.GetAtomWithIdx(bond_atoms[i][0])
                rot_a2 = mol.GetAtomWithIdx(bond_atoms[i][1])
                for neighbor_a1 in rot_a1.GetNeighbors():
                    neighbor_atoms.append(neighbor_a1.GetIdx())
                for neighbor_a2 in rot_a2.GetNeighbors():
                    neighbor_atoms.append(neighbor_a2.GetIdx())
                neighbor_atoms = list(set(neighbor_atoms))
                if all(item in scaffold_atoms for item in neighbor_atoms):
                    remove_index.append(i)
            
    remove_index = list(set(remove_index))
    for index in sorted(remove_index, reverse=True):
        del bond_atoms[index]
    return bond_atoms

def match_num(a_list, b_list):
    n = 0
    for i in a_list:
        for j in b_list:
            if int(i) == int(j):
                n = n + 1
    return n

def PDBQTAtomLines(mol, donors, acceptors):
    """Create a list with PDBQT atom lines for each atom in molecule. Donors
    and acceptors are given as a list of atom indices.
    """

    atom_lines = [line.replace('HETATM', 'ATOM  ')
                  for line in Chem.MolToPDBBlock(mol).split('\n')
                  if line.startswith('HETATM') or line.startswith('ATOM')]

    pdbqt_lines = []
    for idx, atom in enumerate(mol.GetAtoms()):
        pdbqt_line = atom_lines[idx][:56]

        pdbqt_line += '0.00  0.00    '  # append empty vdW and ele
        # Get charge
        charge = 0.
        fields = ['_MMFF94Charge', '_GasteigerCharge', '_TriposPartialCharge']
        for f in fields:
            if atom.HasProp(f):
                charge = atom.GetDoubleProp(f)
                break
        # FIXME: this should not happen, blame RDKit
        if isnan(charge) or isinf(charge):
            charge = 0.
        pdbqt_line += ('%.3f' % charge).rjust(6)

        # Get atom type
        pdbqt_line += ' '
        atomicnum = atom.GetAtomicNum()
        atomhybridization = atom.GetHybridization()
        atombondsnum = atom.GetDegree()
        if atomicnum == 6 and atom.GetIsAromatic():
            pdbqt_line += 'A'
        elif atomicnum == 7 and idx in acceptors:
            pdbqt_line += 'NA'
        elif atomicnum == 8 and idx in acceptors:
            pdbqt_line += 'OA'
        elif atomicnum == 1 and atom.GetNeighbors()[0].GetIdx() in donors:
            pdbqt_line += 'HD'
        elif atomicnum == 1 and atom.GetNeighbors()[0].GetIdx() not in donors:
            pdbqt_line += 'H '
        elif atomicnum == 16 and ( (atomhybridization == Chem.HybridizationType.SP3 and atombondsnum != 4) or atomhybridization == Chem.HybridizationType.SP2 ):
            pdbqt_line += 'SA'
        else:
            pdbqt_line += atom.GetSymbol()
        pdbqt_lines.append(pdbqt_line)
    return pdbqt_lines

def MolToPDBQTBlock(mol, ref, flexible=True, addHs=False, computeCharges=False):
    """Write RDKit Molecule to a PDBQT block

    Parameters
    ----------
        mol: rdkit.Chem.rdchem.Mol
            Molecule with a protein ligand complex
        flexible: bool (default=True)
            Should the molecule encode torsions. Ligands should be flexible,
            proteins in turn can be rigid.
        addHs: bool (default=False)
            The PDBQT format requires at least polar Hs on donors. By default Hs
            are added.
        computeCharges: bool (default=False)
            Should the partial charges be automatically computed. If the Hs are
            added the charges must and will be recomputed. If there are no
            partial charge information, they are set to 0.0.

    Returns
    -------
        block: str
            String wit PDBQT encoded molecule
    """

    # if flexible molecule contains multiple fragments write them separately
    if flexible and len(Chem.GetMolFrags(mol)) > 1:
        return ''.join(MolToPDBQTBlock(frag, flexible=flexible, addHs=addHs, computeCharges=computeCharges)
                       for frag in Chem.GetMolFrags(mol, asMols=True))

    # Identify donors and acceptors for atom typing
    # Acceptors
    patt = Chem.MolFromSmarts('[$([O;H1;v2]),'
                              '$([O;H0;v2;!$(O=N-*),'
                              '$([O;-;!$(*-N=O)]),'
                              '$([o;+0])]),'
                              '$([n;+0;!X3;!$([n;H1](cc)cc),'
                              '$([$([N;H0]#[C&v4])]),'
                              '$([N&v3;H0;$(Nc)])]),'
                              '$([F;$(F-[#6]);!$(FC[F,Cl,Br,I])])]')
    acceptors = list(map(lambda x: x[0], mol.GetSubstructMatches(patt, maxMatches=mol.GetNumAtoms())))
    # Donors
    patt = Chem.MolFromSmarts('[$([N&!H0&v3,N&!H0&+1&v4,n&H1&+0,$([$([Nv3](-C)(-C)-C)]),'
                              '$([$(n[n;H1]),'
                              '$(nc[n;H1])])]),'
                              # Guanidine can be tautormeic - e.g. Arginine
                              '$([NX3,NX2]([!O,!S])!@C(!@[NX3,NX2]([!O,!S]))!@[NX3,NX2]([!O,!S])),'
                              '$([O,S;H1;+0])]')
    donors = list(map(lambda x: x[0], mol.GetSubstructMatches(patt, maxMatches=mol.GetNumAtoms())))
    if addHs:
        mol = Chem.AddHs(mol, addCoords=True, onlyOnAtoms=donors, )
    if addHs or computeCharges:
        AllChem.ComputeGasteigerCharges(mol)

    atom_lines = PDBQTAtomLines(mol, donors, acceptors)
    assert len(atom_lines) == mol.GetNumAtoms()

    pdbqt_lines = []
    pdbqt_lines.append('BEGIN_RES UNL X 1')

    if flexible:
        # Find rotatable bonds
        '''
        rot_bond = Chem.MolFromSmarts('[!$(*#*)&!D1&!$(C(F)(F)F)&'
                                      '!$(C(Cl)(Cl)Cl)&'
                                      '!$(C(Br)(Br)Br)&'
                                      '!$(C([CH3])([CH3])[CH3])&'
                                      '!$([CD3](=[N,O,S])-!@[#7,O,S!D1])&'
                                      '!$([#7,O,S!D1]-!@[CD3]=[N,O,S])&'
                                      '!$([CD3](=[N+])-!@[#7!D1])&'
                                      '!$([#7!D1]-!@[CD3]=[N+])]-!@[!$(*#*)&'
                                      '!D1&!$(C(F)(F)F)&'
                                      '!$(C(Cl)(Cl)Cl)&'
                                      '!$(C(Br)(Br)Br)&'
                                      '!$(C([CH3])([CH3])[CH3])]')
        '''
        #rot_bond = Chem.MolFromSmarts('[!$([NH]!@C(=O))&!D1&!$(*#*)]-&!@[!$([NH]!@C(=O))&!D1&!$(*#*)]') # From Chemaxon
        rot_bond  = Chem.MolFromSmarts('[!$(*#*)&!D1]-&!@[!$(*#*)&!D1]') #single and not ring, really not in ring?
        amide_bonds = Chem.MolFromSmarts('[NX3]-[CX3]=[O,N]') # includes amidines
        tertiary_amide_bonds = Chem.MolFromSmarts('[NX3]([!#1])([!#1])-[CX3]=[O,N]')
        bond_atoms = list(mol.GetSubstructMatches(rot_bond))
        amide_bond_atoms = [(x[0],x[1]) for x in list(mol.GetSubstructMatches(amide_bonds))]
        tertiary_amide_bond_atoms=[(x[0],x[3]) for x in list(mol.GetSubstructMatches(tertiary_amide_bonds))]
        for amide_bond_atom in amide_bond_atoms:
            #bond_atoms.remove(amide_bond_atom)
            amide_bond_atom_reverse=(amide_bond_atom[1],amide_bond_atom[0])
            if amide_bond_atom in bond_atoms:
                bond_atoms.remove(amide_bond_atom)
            elif amide_bond_atom_reverse in bond_atoms:
                bond_atoms.remove(amide_bond_atom_reverse)

        if len(tertiary_amide_bond_atoms) > 0:
            for tertiary_amide_bond_atom in tertiary_amide_bond_atoms:
                bond_atoms.append(tertiary_amide_bond_atom)

        atom_map = get_atom_mapping(mol, ref)
        scaffold_atoms = list(atom_map.keys())
        mol = constrain_embed(mol, ref, atom_map) # embed mol on ref

        # remove rotatable bonds that are not rotatable in the scaffold
        bond_atoms = remove_rot_bond(mol, bond_atoms, scaffold_atoms)


        atom_lines = PDBQTAtomLines(mol, donors, acceptors) # update coordinate

        # Fragment molecule on bonds to get rigid fragments
        bond_ids = [mol.GetBondBetweenAtoms(a1, a2).GetIdx()
                    for a1, a2 in bond_atoms]
                

        if bond_ids:
            for i, b_index in enumerate(bond_ids):
                tmp_frags= Chem.FragmentOnBonds(mol, [b_index], addDummies=False)
                tmp_frags_list=list(Chem.GetMolFrags(tmp_frags))
                #tmp_bigger=0
                if len(tmp_frags_list) == 1:
                    del bond_ids[i]
                    del bond_atoms[i]
                #else:
                #    tmp_bigger= max(len(tmp_frags_list[0]), len(tmp_frags_list[1]))
                #mol.GetBonds()[b_index].SetProp("large_part", str(tmp_bigger))
         #   print(bond_ids)
            mol_rigid_frags = Chem.FragmentOnBonds(mol, bond_ids, addDummies=False)
        else:
            mol_rigid_frags = mol

        #num_torsions = len(bond_atoms)

        frags = list(Chem.GetMolFrags(mol_rigid_frags))

        #list frag  from which bonds ?
        fg_bonds=[]
        fg_num_rotbonds={}
        for fg in frags:
            tmp_bonds=[]
            for a1,a2 in bond_atoms:
                if a1 in fg or a2 in fg:
                    tmp_bonds.append(mol.GetBondBetweenAtoms(a1, a2).GetIdx())
            if tmp_bonds:
                fg_bonds.append(tmp_bonds)
            else:
                fg_bonds.append(None)
            fg_num_rotbonds[fg] = len(tmp_bonds)

        # frag with long branch ?
        fg_bigbranch={}
        for i, fg_bond in enumerate(fg_bonds):
            tmp_bigger=0
            frag_i_mol=frags[i]
            if fg_bond != None: # for rigid mol
                tmp_frags= Chem.FragmentOnBonds(mol, fg_bond, addDummies=False)
                tmp_frags_list=list(Chem.GetMolFrags(tmp_frags))
                for tmp_frag_j in tmp_frags_list:
                    len_tmp_fg_j=len(tmp_frag_j)
                    if frag_i_mol == tmp_frag_j:
                        pass
                    else:
                        if len_tmp_fg_j > tmp_bigger:
                            tmp_bigger=len_tmp_fg_j
            fg_bigbranch[frags[i]] = tmp_bigger

        def weigh_frags(frag):
            return fg_bigbranch[frag], -fg_num_rotbonds[frag],   # bond_weight
        frags = sorted(frags, key=weigh_frags)
       
        match_frag = []
        for i in range(len(frags)):
            x = frags[i]
            match_frag.append(match_num(x, scaffold_atoms))
        pop_num = match_frag.index(max(match_frag))

        # Start writting the lines with ROOT
        pdbqt_lines.append('ROOT')
        frag = frags.pop(pop_num)
        for idx in frag:
            pdbqt_lines.append(atom_lines[idx])
        pdbqt_lines.append('ENDROOT')

        # Now build the tree of torsions usign DFS algorithm. Keep track of last
        # route with following variables to move down the tree and close branches
        branch_queue = []
        current_root = frag
        old_roots = [frag]
        visited_frags = []
        visited_bonds = []
        while len(frags) > len(visited_frags):
            end_branch = True
            for frag_num, frag in enumerate(frags):
                for bond_num, (a1, a2) in enumerate(bond_atoms):
                    if (frag_num not in visited_frags and
                        bond_num not in visited_bonds and
                        (a1 in current_root and a2 in frag or
                         a2 in current_root and a1 in frag)):
                        # direction of bonds is important
                        if a1 in current_root:
                            bond_dir = '%i %i' % (a1 + 1, a2 + 1)
                        else:
                            bond_dir = '%i %i' % (a2 + 1, a1 + 1)
                        pdbqt_lines.append('BRANCH %s' % bond_dir)
                        for idx in frag:
                            pdbqt_lines.append(atom_lines[idx])
                        branch_queue.append('ENDBRANCH %s' % bond_dir)

                        # Overwrite current root and stash previous one in queue
                        old_roots.append(current_root)
                        current_root = frag

                        # remove used elements from stack
                        visited_frags.append(frag_num)
                        visited_bonds.append(bond_num)

                        # mark that we dont want to end branch yet
                        end_branch = False
                        break
                    else:
                        continue
                    break  # break the outer loop as well

            if end_branch:
                pdbqt_lines.append(branch_queue.pop())
                if old_roots:
                    current_root = old_roots.pop()
        # close opened branches if any is open
        while len(branch_queue):
            pdbqt_lines.append(branch_queue.pop())

    else:
        pdbqt_lines.extend(atom_lines)
    pdbqt_lines.append('END_RES UNL X 1')
    return '\n'.join(pdbqt_lines)

mol=Chem.MolFromMolFile(sys.argv[1],removeHs=False)
ref=Chem.MolFromMolFile(sys.argv[2],removeHs=False)
pdbqtlines=MolToPDBQTBlock(mol, ref, True, False, True)
print(pdbqtlines)