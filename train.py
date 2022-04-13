from argparse import ArgumentParser
import model


def declareParserArguments(parser: ArgumentParser) -> ArgumentParser:
    # Add arguments
    parser.add_argument(
        "--data_informations",
        action="store_true",
        default=False,
        help="Information about books.",
    )

    parser.add_argument(
        "--visualize_results",
        action="store_true",
        default=False,
        help="Whether to visualize results",
    )

    return parser.parse_args()


if __name__ == "__main__":
    # Declare an ArgumentParser object
    parser = ArgumentParser(description="TextGenerator")
    # Add the arguments
    args = declareParserArguments(parser=parser)

    # Create a KerasTrainer object
    trainer = model.KerasTrainer(args=args)
    # Initiate the training
    trainer._train()
    # Result 100 Epoch: loss=1.2206 acc:0.7534
