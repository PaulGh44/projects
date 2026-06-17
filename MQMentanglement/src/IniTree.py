from anytree import Node, RenderTree


def iniTree(a: float, listcosmo: list[float], liststates: list[str], listmodes: list[str]):
    root = Node("Entropy in c=1")

    lattice_node = Node(f"Lattice parameter a={a}", parent=root, a=a)

    for sector in ["Singlet", "Non-singlet"]:
        sector_node = Node(sector, parent=lattice_node, sector=sector)

        for truncation in ["Genus 0 minisuperspace","Genus 0"]:
            truncation_node = Node(truncation, parent=sector_node, truncation=truncation, sector=sector)

            for state in liststates:
                state_node = Node(state, parent=truncation_node, state=state, truncation=truncation, sector=sector)

                for mode in listmodes:
                    mode_node = Node(mode, parent=state_node, mode=mode, state=state, truncation=truncation, sector=sector)

                    for mu in listcosmo:
                        mu_node = Node(f"Cosmological constant mu={mu}", parent=mode_node, mu=mu, mode=mode, state=state, truncation=truncation, sector=sector, file=None)

    return root


# root = iniTree(a=0.05, listcosmo=[0.0, 0.1], liststates=["vacuum", "thermal", "TFD"], listmodes=["fixed_left", "fixed_right", "moving_center"])

# for pre, _, node in RenderTree(root):
#     print(f"{pre}{node.name}")