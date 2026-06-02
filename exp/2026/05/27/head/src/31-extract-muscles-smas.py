from pathlib import Path

import pyvista as pv

from liblaf import cherries, melon


class Config(cherries.BaseConfig):
    input: Path = cherries.input("30-muscles.m.vtkhdf")
    output: Path = cherries.output("31-muscles-smas.m.vtkhdf")


SMAS_MUSCLES: tuple[str, ...] = (
    "Occipitofrontalis epicranius001",
    "Temporal fascia001",
    "Platysma001",
    "Orbicularis oris001",
    "Orbicularis oculi001",
    "Zygomaticus major001",
    "Zygomaticus minor001",
    "Risorius001",
    "Levator labii superioris alaeque nasi001",
    "Levator labii superioris001",
    "Levator anguli oris001",
    "Depressor anguli001",
    "Depressor labii inferioris001",
    "Mentalis001",
    "Nasalis transverse portion001",
    "Nasalis alarportion001",
    "Depressor septi001",
    "Procerus001",
    "Corrugator supercilii001",
    "Depressor supercilli001",
    "Auricularis anterior001",
    "Auricularis posterior001",
    "Auricularis superior001",
)


def main(cfg: Config) -> None:
    muscles: pv.MultiBlock = melon.io.load_multiblock(cfg.input)
    output: pv.MultiBlock = pv.MultiBlock()
    for idx, block in enumerate(muscles):
        name: str = muscles.get_block_name(idx)
        if name.startswith(SMAS_MUSCLES):
            output.append(block, name)
    output.save(cfg.output)


if __name__ == "__main__":
    cherries.main(main)
