from argparse import ArgumentParser


def declareParserArguments(parser: ArgumentParser) -> ArgumentParser:
    # Add arguments
    parser.add_argument(
        "--data_informations",
        action="store_true",
        default=False,
        help="Information about books.",
    )

    return parser.parse_args()
