# Credit: TAMU CSCE 642, Dr. Guni Sharon for the format of this file

solvers = [
    "random",
    "greedy",
    "epsilon_greedy",
    "a2c",
    "a2c_elig"
]


def get_solver_class(name):
    if name == solvers[0]:
        from Solvers.RandomAgent import RandomAgent # so there's no name confusion/conflict with other randoms from the libraries
        return RandomAgent # should only be used for testing
    elif name == solvers[1]:
        from Solvers.GreedyAgent import GreedyAgent
        return GreedyAgent
    elif name == solvers[2]:
        from Solvers.EpiGreedyAgent import EpiGreedyAgent
        return EpiGreedyAgent
    elif name == solvers[3]:
        from Solvers.A2CAgent import A2CAgent
        return A2CAgent
    elif name == solvers[4]:
        from Solvers.A2CEligibility import A2CEligibility
        return A2CEligibility
    else:
        assert False, "unknown solver name {}. solver must be from {}".format(
            name, str(solvers)
        )
