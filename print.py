import argparse

import numpy as np

from mtgproxies import fetch_scans_scryfall, print_cards_fpdf, print_cards_matplotlib
from mtgproxies.cli import parse_decklist_spec


def papersize(string: str) -> np.ndarray:
    spec = string.lower()
    if spec == "a4":
        return np.array([21, 29.7]) / 2.54
    if "x" in spec:
        split = spec.split("x")
        return np.array([float(split[0]), float(split[1])])
    raise argparse.ArgumentTypeError()

def genPdf(customArgs = None, pipe=None):
    parser = argparse.ArgumentParser(description="Prepare a decklist for printing.")
    parser.add_argument(
        "decklist",
        metavar="decklist_spec",
        help="path to a decklist in text/arena format, or manastack:{manastack_id}, or archidekt:{archidekt_id}",
    )
    parser.add_argument("outfile", help="output file. Supports pdf, png and jpg.")
    parser.add_argument("--dpi", help="dpi of output file (default: %(default)d)", type=int, default=300)
    parser.add_argument(
        "--paper",
        help="paper size in inches or preconfigured format (default: %(default)s)",
        type=papersize,
        default="a4",
        metavar="WIDTHxHEIGHT",
    )
    parser.add_argument(
        "--scale",
        help="scaling factor for printed cards (default: %(default)s)",
        type=float,
        default=1.0,
        metavar="FLOAT",
    )
    parser.add_argument(
        "--border_crop",
        help="how much to crop inner borders of printed cards (default: %(default)s)",
        type=int,
        default=14,
        metavar="PIXELS",
    )
    parser.add_argument(
        "--background",
        help='background color, either by name or by hex code (e.g. black or "#ff0000", default: %(default)s)',
        type=str,
        default=None,
        metavar="COLOR",
    )
    parser.add_argument(
        "--intelligent_background",
        action=argparse.BooleanOptionalAction,
        help='color background based on individual card',
        default=False,
    )
    parser.add_argument(
        "--directInput",
        action=argparse.BooleanOptionalAction,
        help='instead of filepath to text file, is the input string directly',
        default=False,
    )
    parser.add_argument(
        "--directOutput",
        action=argparse.BooleanOptionalAction,
        help='instead of saving the file locally, is the output is returned directly',
        default=False,
    )
    parser.add_argument(
        "--hSpace",
        help="horizontal spacing between cards (default: %(default)s)",
        type=int,
        default=0,
        metavar="PIXELS",
    )
    parser.add_argument(
        "--vSpace",
        help="vertical spacing between cards (default: %(default)s)",
        type=int,
        default=0,
        metavar="PIXELS",
    )
    parser.add_argument("--cropmarks", action=argparse.BooleanOptionalAction, default=True, help="add crop marks")

    if(customArgs == None):
        args = parser.parse_args()
    else:
        args = parser.parse_args(customArgs)

    # Parse decklist
    decklist = parse_decklist_spec(args.decklist, inputAsString=args.directInput, pipe=pipe)

    # Fetch scans
    images = fetch_scans_scryfall(decklist, pipe=pipe)

    # Plot cards
    if args.outfile.endswith(".pdf"):
        import matplotlib.colors as colors

        background_color = args.background
        if background_color is not None:
            background_color = (np.array(colors.to_rgb(background_color)) * 255).astype(int)

        pdf = print_cards_fpdf(
            images,
            args.outfile,
            papersize=args.paper * 25.4,
            cardsize=np.array([2.5, 3.5]) * 25.4 * args.scale,
            border_crop=args.border_crop,
            hSpace=args.hSpace,
            vSpace=args.vSpace,
            background_color=background_color,
            intelligent_background=args.intelligent_background,
            cropmarks=args.cropmarks,
            return_pdf=args.directOutput,
            pipe=pipe
        )

        if(args.directOutput == True):
            return pdf
    else:
        print_cards_matplotlib(
            images,
            args.outfile,
            papersize=args.paper,
            cardsize=np.array([2.5, 3.5]) * args.scale,
            dpi=args.dpi,
            border_crop=args.border_crop,
            background_color=args.background,
        )

if __name__ == "__main__":
    genPdf()

