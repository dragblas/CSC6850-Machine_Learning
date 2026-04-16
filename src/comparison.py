"""
  compares SMILES to compute accuracy, both exact string match and chemical equivalence match
"""
from rdkit import Chem

def match(prediction, target):
  """
  computes match accuracy between predicted and target SMILES (from list of strings)
  """

  if len(prediction) != len(target):
    raise ValueError("Length mismatch")

  results = []
  correct = 0

  for p, t in zip(prediction, target):
    if p.strip() == t.strip():
      results.append(1)
      correct += 1
    else:
      results.append(0)

  accuracy = correct / len(prediction)

  return accuracy, results # accuracy is the total accuracy, results is a list of 1s and 0s indicating which predictions were correct

# normalizes a SMILES string using RDKit
def canonicalize(smiles):
  mol = Chem.MolFromSmiles(smiles)
  if mol is None:
    return None
  return Chem.MolToSmiles(mol, canonical=True)

# computes chemical equivalence match between predicted and target SMILES (from list of strings)
# This is for cases where SMILES might not give the same string, but have the same chemical/molecules 
def canonical_match(predictions, targets):
  """
  
  """
  if len(predictions) != len(targets):
    raise ValueError("Length mismatch")

  results = []
  correct = 0

  for p, t in zip(predictions, targets):
    p_can = canonicalize(p)
    t_can = canonicalize(t)

    if p_can is not None and p_can == t_can:
      results.append(1)
      correct += 1
    else:
      results.append(0)

  return correct / len(predictions), results
