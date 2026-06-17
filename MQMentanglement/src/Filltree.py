import numpy as np
from EEconstructor import entropy_as_function_of_left_endpoint, entropy_as_function_of_right_endpoint, entropy_as_function_of_center_point
from anytree import Node, RenderTree
from IniTree import iniTree
from pathlib import Path

def attach_entropy_file(node, entropy_array, filename):
    """
    This simple function allows to attach a file "filename" to a given node of the tree from a numpy list of entropies entropy_array
    """
    filename = Path(filename)
    filename.parent.mkdir(parents=True, exist_ok=True)

    np.savetxt(filename, entropy_array)
    node.file = filename

# ------------------------------------------------------------
# Simple test
# ------------------------------------------------------------
if __name__ == "__main__":
    root = iniTree(a=0.1, listcosmo=[0.1], liststates=["vacuum"], listmodes=["fixed_right_endpoint"])

    # Print tree
    for pre, _, node in RenderTree(root):
        print(f"{pre}{node.name}")

    # Pick one leaf of the tree
    leaf = root.leaves[0]

    print("\nSelected leaf:")
    print(leaf)
    print("mu =", leaf.mu)
    print("state =", leaf.state)
    print("mode =", leaf.mode)

    # Fake entropy data
    entropy_array = np.array([
        [0.0, 0.12],
        [0.1, 0.18],
        [0.2, 0.24],
        [0.3, 0.29],
    ])

    # Attach txt file to the leaf
    filename = (
        Path("data")
        / f"a_{root.children[0].a}"
        / leaf.sector
        / leaf.truncation.replace(" ", "_")
        / leaf.state
        / leaf.mode
        / f"mu_{leaf.mu}.txt"
    )

    attach_entropy_file(leaf, entropy_array, filename)

    print("\nFile attached to leaf:")
    print(leaf.file)

    # Check that the file exists
    print("\nDoes file exist?")
    print(leaf.file.exists())

    # Reload the data
    loaded_array = np.loadtxt(leaf.file)

    print("\nReloaded data:")
    print(loaded_array)
