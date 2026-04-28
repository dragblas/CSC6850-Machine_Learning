"""
Evaluation helpers for exact and canonical SMILES matching.
"""


def exact_match(predictions, targets):
    """Return exact string-match accuracy and a per-sample correctness list."""
    if len(predictions) != len(targets):
        raise ValueError("Length mismatch")
    if len(predictions) == 0:
        return 0.0, []

    results = [
        1 if prediction.strip() == target.strip() else 0
        for prediction, target in zip(predictions, targets)
    ]
    return sum(results) / len(results), results


def canonicalize(smiles):
    """Convert a SMILES string to RDKit canonical SMILES, or None if invalid."""
    from rdkit import Chem, RDLogger

    RDLogger.DisableLog("rdApp.*")

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True)


def canonical_match(predictions, targets):
    """Return chemical-equivalence accuracy after RDKit canonicalization."""
    if len(predictions) != len(targets):
        raise ValueError("Length mismatch")
    if len(predictions) == 0:
        return 0.0, []

    results = []

    for prediction, target in zip(predictions, targets):
        prediction_canonical = canonicalize(prediction)
        target_canonical = canonicalize(target)
        results.append(
            1
            if prediction_canonical is not None
            and prediction_canonical == target_canonical
            else 0
        )

    return sum(results) / len(results), results


def valid_smiles_rate(smiles_list):
    """Return the fraction of generated SMILES strings RDKit can parse."""
    if len(smiles_list) == 0:
        return 0.0, []

    results = [1 if canonicalize(smiles) is not None else 0 for smiles in smiles_list]
    return sum(results) / len(results), results


def match(predictions, targets):
    return exact_match(predictions, targets)
