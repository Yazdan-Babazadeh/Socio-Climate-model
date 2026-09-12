"""Command-line entry point for the socio-climate manuscript figures."""

import argparse

from figures import (
    figure_1,
    figure_2_mitigation_cost,
    figure_2_social_learning,
    figure_2_social_norms,
    figure_3,
    figure_4,
    figure_5,
    figure_6,
    generate_all_figures,
)


FIGURE_FUNCTIONS = {
    "1": figure_1,
    "2-k": figure_2_social_learning,
    "2-beta": figure_2_mitigation_cost,
    "2-delta": figure_2_social_norms,
    "3": figure_3,
    "4": figure_4,
    "5": figure_5,
    "6": figure_6,
}


def main():
    parser = argparse.ArgumentParser(
        description="Generate socio-climate model manuscript figures."
    )
    parser.add_argument(
        "--figure",
        choices=["all", *FIGURE_FUNCTIONS],
        default="all",
        help="Figure or parameter-sweep panel to generate.",
    )
    args = parser.parse_args()

    if args.figure == "all":
        generate_all_figures()
    else:
        FIGURE_FUNCTIONS[args.figure]()


if __name__ == "__main__":
    main()
