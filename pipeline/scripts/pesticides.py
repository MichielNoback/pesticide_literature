
pesticide_classes = None
pesticide_terms = None


def _create_pesticide_classes():
    """
    Create a list of pesticide classes and their corresponding names.
    """
    global pesticide_classes
    # Define the pesticide classes
    pesticide_classes = ['pesticide', 'acaricide', 'algicide', 'avicide', 'bactericide', 'fungicide',
                         'herbicide', 'insecticide', 'molluscicide', 'nematicide', 'rodenticide', 'virucide',
                         'miticide']
    
    # Create a list of pesticide classes with "al" suffix
    pesticide_classes2 = [(p[:-1] + "al") for p in pesticide_classes]
    
    # Combine both lists
    pesticide_classes.extend(pesticide_classes2)
    

pesticide_names = ["acetochlor",
    "alachlor",
    "aldicarb",
    "aldrin",
    "amitraz",
    "atrazine",
    "benomyl",
    "benzalkonium chloride",
    "bifenthrin",
    "carbaryl",
    "carbendazim",
    "carbofos",
    "carbofuran",
    "captan",
    "chlordane",
    "chlordecone",
    "chlordimeform",
    "chlormequat",
    "chlorcholine chloride",
    "chlorothalonil",
    "chlorpyrifos",
    "cyfloxylate",
    "cyfluthrin",
    "cyhalothrin", #lamda-cyhalothrin
    "cypermethrin",
    "deltamethrin",
    "demeton",
    "dichlorovos",
    "dieldrin",
    "diethofencarb",
    "dimetoat",
    "diquat",
    "endosulfan",
    "epn", #nitrophenyl phenylphosphonothionate
    "fenitrothion",
    "fenobucarb",
    "fenthion",
    "fenvalarate",
    "fipronil",
    "flucythrinate",
    "glufosinate",
    "glyphosate",
    "imidacloprid",
    "lindane",
    "malathion",
    "methiocarb",
    "methoxychlor",
    "metolachlor",
    "metribuzin",
    "molinate",
    "monocrotophos",
    "oxamyl",
    "parathion",
    "pentachlorophenol",
    "permethrin",
    "profenofos",
    "propanil",
    "propoxur",
    "pyrethroid",
    "sulprofos",
    "temephos",
    "tetrahydrophthalimide",
    "thiobencarb",
    "trichlorfon",
    "trichlorofon",
    "trifluralin",
    "vinclozolin"]

def _create_pesticide_terms():
    """
    Create a list of pesticide terms.
    """
    global pesticide_terms
    # Combine the classes and names into a single list
    pesticide_terms = pesticide_classes + pesticide_names

_create_pesticide_classes()
_create_pesticide_terms()

if __name__ == "__main__":
    # Test the function
    print("Pesticide classes:")
    print(pesticide_classes)
    print("\nPesticide terms:")
    print(pesticide_terms)