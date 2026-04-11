def stage_explanation(stage_text: str, kernel: str) -> str:
    if stage_text.startswith("1."):
        return (
            "Most csak a pontokat látod. Itt még nincs megmutatva, "
            "hogyan választja el őket az SVM."
        )
    if stage_text.startswith("2."):
        return (
            "A háttér már a modell által becsült döntési régiókat mutatja: "
            "a sík mely részeit melyik osztályhoz rendeli."
        )
    if stage_text.startswith("3."):
        return (
            "A fekete görbe vagy vonal a döntési határ: itt vált a modell "
            "egyik osztályról a másikra."
        )
    if kernel == "linear":
        return (
            "A szürke vonalak a marginok, a kiemelt pontok a support vectorok. "
            "Ezek határozzák meg a maximális margójú elválasztót."
        )
    return (
        "Nemlineáris kernel esetén a döntési határ görbült lehet, "
        "de a support vectorok itt is kulcsszerepet játszanak."
    )


def kernel_label(kernel: str) -> str:
    return {
        "linear": "lineáris",
        "poly": "polinomiális",
        "rbf": "RBF",
    }[kernel]


def short_kernel_explanation(kernel: str) -> str:
    if kernel == "linear":
        return "Jó választás, ha az adatok közel egyenessel elválaszthatók."
    if kernel == "poly":
        return (
            "Polinomiális kölcsönhatásokat is figyelembe vesz, "
            "ezért görbült határokat tud tanulni."
        )
    return (
        "Nagyon rugalmas, lokális hasonlóság alapján "
        "görbült döntési határt épít."
    )