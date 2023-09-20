from typing import Optional
from bochemian.gprotorch.data_featuriser.featurisation import one_hot, drfp, fingerprints, rxnfp, rxnfp2, gpt2, drxnfp
from bochemian.gprotorch.data_featuriser.featurisation import fingerprints, fragments, mqn_features, cddd, xtb, chemberta_features, bag_of_characters, graphs, random_features
from bochemian.gprotorch.dataloader import DataLoader
import numpy as np
import pandas as pd
from rdkit import Chem


def canonicalize(smiles):
    return Chem.MolToSmiles(Chem.MolFromSmiles(smiles), isomericSmiles=True)

def generate_canonical_reaction_smiles(row, product_column='product'):
    reactant_columns = [col for col in row.index if col not in ['product', 'objective']]
    
    reactants = []
    for col in reactant_columns:
        reactant_smiles = canonicalize(row[col])
        reactants.append(reactant_smiles)
        
    reactants_str = '.'.join(reactants)
    product = canonicalize(row[product_column])
    reaction_smiles = f"{reactants_str}>>{product}"
    
    return reaction_smiles


class DataFeaturizer(DataLoader):
    def __init__(self, representation, bond_radius=3, n_bits=2048, graphein_config=None):
        self.representation = representation
        self.bond_radius = bond_radius
        self.n_bits = n_bits
        self.graphein_config = graphein_config

        self._features = None
        self._labels = None

    @property
    def features(self):
        return self._features

    @features.setter
    def features(self, value):
        self._features = value

    @property
    def labels(self):
        return self._labels

    @labels.setter
    def labels(self, value):
        self._labels = value
    
    def validate(self, drop=True):
        invalid_idx = []

    def set_features(self, task, design_space: pd.DataFrame):
        self.task = task
        if task == "reaction_optimization":
            if self.representation in ['drfp', 'rxnfp']:
                if not "rxn" in design_space.columns:
                    self.features = self._generate_reaction_smiles(design_space)
                else: 
                    self.features = design_space['rxn']
            else:
                self.features = design_space
        
        elif task == "molecular_optimization":
            mol_col = list(set(design_space.columns) -  set(["objective"]))
            self.features = design_space[mol_col[0]]


    def _generate_reaction_smiles(self, design_space: pd.DataFrame):
        # Extract the reaction components columns
        rxns = design_space.apply(lambda x: generate_canonical_reaction_smiles(x), axis=1).tolist()
        return rxns


    def featurize(self):
        if self.task == "molecular_optimization":
            valid_representations = ["fingerprints",
                                    "fragments",
                                    "fragprints",
                                    "bag_of_smiles",
                                    "bag_of_selfies",
                                    "chemberta",
                                    "graphs",
                                    "chemprints",
                                    "mqn",
                                    "cddd",
                                    "xtb",
                                    "cddd+xtb",
                                    "mqn+xtb",
                                    "cddd+xtb+mqn",
                                    "fingerprints+xtb",
                                    "fragprints+xtb"]
            
        elif self.task == "reaction_optimization":
            valid_representations = ["ohe",
                                    "rxnfp",
                                    "rxnfp2",
                                    "drfp",
                                    "drxnfp",
                                    "bag_of_smiles",
                                    "gpt2"]

        if self.representation not in valid_representations:
            raise ValueError(f"Invalid self.representation: {self.representation}")

        if self.representation == "ohe":
            self.features = one_hot(self.features)

        elif self.representation == "rxnfp":
            self.features = rxnfp(self.features.to_list())

        elif self.representation == "rxnfp2":
            self.features = rxnfp2(self.features.to_list())

        elif self.representation == "drfp":
            print(self.features)
            self.features = drfp(
                self.features.to_list(), nBits=self.n_bits, bond_radius=self.bond_radius
            )
        elif self.representation == "gpt2":
            self.features == gpt2(self.features.to_list())

        elif self.representation == "drxnfp":
            self.features = drxnfp(
                self.features.to_list(), bond_radius=self.bond_radius, nBits=self.n_bits
            )

        if self.representation == "fingerprints":
            self.features = fingerprints(
                self.features, bond_radius=self.bond_radius, nBits=self.n_bits
            )

        elif self.representation == "fragments":
            self.features = fragments(self.features)

        elif self.representation == "fragprints":
            self.features = np.concatenate(
                (
                    fingerprints(self.features, bond_radius=self.bond_radius, nBits=self.n_bits),
                    fragments(self.features),
                ),
                axis=1,
            )
        elif self.representation == "chemprints":
            self.features = np.concatenate(
                (
                    chemberta_features(self.features),
                    fingerprints(self.features, bond_radius=self.bond_radius, nBits=self.n_bits),
                ),
                axis=1,
            )
        elif self.representation == "mqn":
            self.features = mqn_features(self.features)

        elif self.representation == "cddd":
            self.features = cddd(self.features)

        elif self.representation == "xtb":
            self.features = xtb(self.features)

        elif self.representation == "cddd+xtb":
            self.features = np.concatenate(
                (
                    cddd(self.features),
                    xtb(self.features),
                ),
                axis=1,
            )

        elif self.representation == "cddd+xtb+mqn":
            self.features = np.concatenate(
                (
                    cddd(self.features),
                    xtb(self.features),
                    mqn_features(self.features),
                ),
                axis=1,
            )

        elif self.representation == "mqn+xtb":
            self.features = np.concatenate(
                (
                    mqn_features(self.features),
                    xtb(self.features),
                ),
                axis=1,
            )

        elif self.representation == "fingerprints+xtb":
            self.features = np.concatenate(
                (
                    fingerprints(self.features, bond_radius=self.bond_radius, nBits=self.n_bits),
                    xtb(self.features),
                ),
                axis=1,
            )

        elif self.representation == "fragprints+xtb":
            self.features = np.concatenate(
                (
                    fingerprints(self.features, bond_radius=self.bond_radius, nBits=self.n_bits),
                    fragments(self.features),
                    xtb(self.features),
                ),
                axis=1,
            )

        elif self.representation == "bag_of_selfies":
            self.features = bag_of_characters(self.features, selfies=True)

        elif self.representation == "bag_of_smiles":
            self.features = bag_of_characters(self.features)

        elif self.representation == "random":
            self.features = random_features(self.features)

        elif self.representation == "chemberta":
            self.features = chemberta_features(self.features)

        elif self.representation == "chemberta+xtb":
            self.features = np.concatenate(
                (
                    chemberta_features(self.features),
                    xtb(self.features),
                ),
                axis=1,
            )

        elif self.representation == "chemberta+xtb+mqn":
            self.features = np.concatenate(
                (
                    chemberta_features(self.features),
                    xtb(self.features),
                    mqn_features(self.features),
                ),
                axis=1,
            )

        elif self.representation == "graphs":
            self.features = graphs(self.features, self.graphein_config)

    def load_benchmark(self, benchmark, path):
        """Loads features and labels from one of the included benchmark datasets
                and feeds them into the DataLoader.

        :param benchmark: the benchmark dataset to be loaded, one of
            ``[Additives with molecular or reaction smiles (rxn)]``
        :type benchmark: str
        :param path: the path to the dataset in csv format
        :type path: str
        """


        benchmark_structure = {
            "additives": {
                "features": ["additives"],
                "labels": "objective",
            },
            "additives_rxn": {
                "features": ["rxn"],
                "labels": "objective",
            },
        }

        benchmarks = {
            f"{key}_plate_{i}": value
            for key, value in benchmark_structure.items()
            for i in range(1,5)
        }
           

        if benchmark not in benchmarks.keys():
            raise Exception(
                f"The specified benchmark choice ({benchmark}) is not a valid option. "
                f"Choose one of {list(benchmarks.keys())}."
            )

        else:
            df = pd.read_csv(path)
            # drop nans from the datasets
            nans = df[benchmarks[benchmark]["labels"]].isnull().to_list()
            nan_indices = [nan for nan, x in enumerate(nans) if x]
            self.features = df[benchmarks[benchmark]["features"]].drop(nan_indices)
            self.labels = (
                df[benchmarks[benchmark]["labels"]].dropna().to_numpy().reshape(-1, 1)
            )

