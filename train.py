import os

from attention_matching.args import get_train_args_parser, validate_train_args
from attention_matching.pipeline import main

if __name__ == "__main__":

    parser = get_train_args_parser()

    args, unknown = parser.parse_known_args()

    if len(unknown) > 0:
        print("UNKNOWN ARGS: ", unknown)
    print("ARGS: ", args)

    args = validate_train_args(args)

    args.path_model = os.path.join(args.path_model, args.run_name)

    main(args)
