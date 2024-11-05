# Credit: TAMU CSCE 642, Dr. Guni Sharon for the format of this file

domains = [
    "Empty",
    "SingleDuck",
    "City",
    "WindyCity",
]


def get_domain_class(name, render_mode=""):
    if name == domains[0]:
        from Domains.Empty import Empty 
        return Empty # should only be used for testing
    elif name == domains[1]:
        from Domains.SingleDuck import SingleDuck
        return SingleDuck
    elif name == domains[2]:
        from Domains.City import City
        return City
    elif name == domains[3]:
        from Domains.WindyCity import WindyCity
        return WindyCity
    else:
        assert False, "unknown domain name {}. domain must be from {}".format(
            name, str(domains)
        )
