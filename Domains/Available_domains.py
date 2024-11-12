# Credit: TAMU CSCE 642, Dr. Guni Sharon for the format of this file

domains = [
    "Empty",
    "SingleDuck",
    "City",
    "WindyCity",
]


def get_domain_class(name, render_mode=""):
    if name == domains[0]:
        from Domains.Empty import EmptyDomain 
        return EmptyDomain # should only be used for testing
    elif name == domains[1]:
        from Domains.SingleDuck import SingleDuckDomain
        return SingleDuckDomain
    elif name == domains[2]:
        from Domains.City import CityDomain
        return CityDomain
    elif name == domains[3]:
        from Domains.WindyCity import WindyCityDomain
        return WindyCityDomain
    else:
        assert False, "unknown domain name {}. domain must be from {}".format(
            name, str(domains)
        )
