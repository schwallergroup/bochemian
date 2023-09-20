import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors
import pubchempy as pcp
import pandas as pd


# Define the function to get component details
def get_component_details(smiles):
    try:
        compound = pcp.get_compounds(smiles, "smiles")[0]
        molecule = Chem.MolFromSmiles(smiles)
        details = {
            "iupac_name": compound.iupac_name,
            "molecular_formula": compound.molecular_formula,
            "molecular_weight": compound.molecular_weight,
            "xlogp": compound.xlogp,
            "tpsa": compound.tpsa,
            "heavy_atom_count": Descriptors.HeavyAtomCount(molecule),
            "h_acceptor_count": Descriptors.NumHAcceptors(molecule),
            "h_donor_count": Descriptors.NumHDonors(molecule),
            "rotatable_bond_count": Descriptors.NumRotatableBonds(molecule),
            "ring_count": Descriptors.RingCount(molecule),
            "complexity": compound.complexity,
        }
    except:
        details = {
            "iupac_name": "Not available",
            "molecular_formula": "Not available",
            "molecular_weight": "Not available",
            "xlogp": "Not available",
            "tpsa": "Not available",
            "heavy_atom_count": "Not available",
            "h_acceptor_count": "Not available",
            "h_donor_count": "Not available",
            "rotatable_bond_count": "Not available",
            "ring_count": "Not available",
            "complexity": "Not available",
        }
    return details


# Generate procedures for each reaction
def generate_procedure(row, components, details, procedure_template, smiles_details):
    component_details_text = []
    for component in components:
        component_details = smiles_details[row[component]]
        component_text = [f"- {component.capitalize()}: {row[component]}"]
        if details is None:
            details = []
        for detail in details:
            component_text.append(
                f"  - {detail.replace('_', ' ').capitalize()}: {component_details[detail]}"
            )
        component_details_text.append("\n".join(component_text))
    procedure = procedure_template.format(
        component_details="\n".join(component_details_text)
    )
    return procedure

