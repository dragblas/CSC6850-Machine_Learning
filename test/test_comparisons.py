"""
tests exact vs canonical match comparison functions 
"""

from src.comparison import match, canonical_match

predictions = [
    "CCO",
    "c1ccccc1",
    "OCC",      # same as CCO chemically
    "WRONG",
    "CCOCC"
]

targets = [
    "CCO",
    "c1ccccc1",
    "CCO",
    "CCN",
    "CCOCC"
]

acc_exact, res_exact = match(predictions, targets)
acc_can, res_can = canonical_match(predictions, targets)

print("Exact Match:", acc_exact)
print("Exact Results:", res_exact)

print("Canonical Match:", acc_can)
print("Canonical Results:", res_can)
