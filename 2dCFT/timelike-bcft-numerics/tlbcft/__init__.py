"""tlbcft: numerical definitions and consistency checks of Liouville boundary CFT at c <= 1 (timelike Liouville),
accompanying "Liouville boundary conformal field theory at c <= 1" (Tsiares, Betzios, Papadoulaki, Ghiringhelli).

Modules
  special   Barnes G, Barnes double Gamma Gamma_b (rational b^2 and arbitrary b), double sine S_b
  kernels   c >= 25 fusion and modular kernels F, S; their Virasoro-Wick rotations RF, RS
  blocks    sphere four-point blocks (elliptic form), torus one-point blocks, five-point comb blocks
  theory    the bulk and boundary data of Table I (c <= 1) and of the c >= 25 baseline
  numerics  contour grids on eps + iR, quadrature rules and circle residues
"""
