import argparse


def get_args_for_encoder_training():
    # Define options
    parser = argparse.ArgumentParser(description="Template")

    # Dataset options

    ### BLOCK DESIGN ###
    # Data
    parser.add_argument(
        "--eeg_dataset", default=None, help="EEG dataset path"
    )  # 5-95Hz
    parser.add_argument("--image_dir", default=None, help="ImageNet dataset path")
    # Splits
    parser.add_argument(
        "--splits_path", default=None, help="splits path"
    )  # All subjects
    ### BLOCK DESIGN ###
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Directory to save the model checkpoints and logs.",
    )

    parser.add_argument("--clip_model", default="openai/clip-vit-base-patch32")

    parser.add_argument(
        "-sn", "--split_num", default=0, type=int, help="split number"
    )  # leave this always to zero.

    # Subject selecting
    parser.add_argument(
        "-sub",
        "--subject",
        default=0,
        type=int,
        help="choose a subject from 1 to 6, default is 0 (all subjects)",
    )

    # Time options: select from 20 to 460 samples from EEG data
    parser.add_argument(
        "-tl", "--time_low", default=20, type=float, help="lowest time value"
    )
    parser.add_argument(
        "-th", "--time_high", default=460, type=float, help="highest time value"
    )
    # Training options
    parser.add_argument("--save_every", type=int, default=5)

    parser.add_argument("--device", type=str, default="cuda")

    # train args

    parser.add_argument(
        "--batch_size", type=int, default=16, help="Batch size for training."
    )
    parser.add_argument(
        "--num_epochs", type=int, default=100, help="Number of epochs for training."
    )
    parser.add_argument(
        "--save_steps",
        default=5000,
        type=int,
        help="Number of steps between saving checkpoints.",
    )
    parser.add_argument(
        "--logging_steps", default=30, type=int, help="Number of steps between logging."
    )
    parser.add_argument(
        "--learning_rate",
        default=2e-5,
        type=float,
        help="The initial learning rate for Adam.",
    )
    parser.add_argument(
        "--optim",
        default="adamw_hf",
        type=str,
        help="Optimizer to use for training.",
    )
    parser.add_argument(
        "--weight_decay", default=0.001, type=float, help="Weight decay to apply."
    )
    parser.add_argument(
        "--max_grad_norm",
        default=0.3,
        type=float,
        help="Max gradient norm to clip gradients.",
    )
    parser.add_argument(
        "--max_steps",
        default=-1,
        type=int,
        help="If > 0: set total number of training steps to perform.",
    )
    parser.add_argument(
        "--warmup_ratio",
        default=0.3,
        type=float,
        help="Ratio of total steps to perform linear learning rate warmup.",
    )
    parser.add_argument(
        "--group_by_length",
        action="store_true",
        help="Whether to group samples of roughly the same length together.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        default="constant",
        type=str,
        help="Type of learning rate scheduler.",
    )
    # Parse arguments
    args = parser.parse_args()
    return args


def get_args_for_llm_finetuning():
    # Define options
    parser = argparse.ArgumentParser(description="Template")

    # Dataset options

    ### BLOCK DESIGN ###
    # Data
    parser.add_argument(
        "--eeg_dataset", default=None, help="EEG dataset path"
    )  # 5-95Hz
    parser.add_argument("--image_dir", default=None, help="ImageNet dataset path")
    # Splits
    parser.add_argument(
        "--splits_path", default=None, help="splits path"
    )  # All subjects
    ### BLOCK DESIGN ###
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Directory to save the model checkpoints and logs.",
    )
    parser.add_argument(
        "--llm_backbone_name_or_path",
        type=str,
        default="",
        help="Name or path of the image tower model.",
    )
    parser.add_argument(
        "--load_in_8bit", default=False, help="load LLM in 8 bit", action="store_true"
    )
    parser.add_argument(
        "--use_lora", default=False, help="load LLM in 8 bit", action="store_true"
    )
    parser.add_argument(
        "--no_stage2", default=False, help="Directly begin stage3", action="store_true"
    )
    parser.add_argument(
        "--eeg_encoder_path",
        type=str,
        required=True,
        help="Path to the fine-tuned EEG encoder",
    )
    parser.add_argument(
        "--saved_pretrained_model_path",
        type=str,
        default="/tmp",
        help="Directory to load the model checkpoints",
    )
    parser.add_argument("--clip_model", default="openai/clip-vit-base-patch32")

    parser.add_argument(
        "-sn", "--split_num", default=0, type=int, help="split number"
    )  # leave this always to zero.

    # Subject selecting
    parser.add_argument(
        "-sub",
        "--subject",
        default=0,
        type=int,
        help="choose a subject from 1 to 6, default is 0 (all subjects)",
    )

    # Time options: select from 20 to 460 samples from EEG data
    parser.add_argument(
        "-tl", "--time_low", default=20, type=float, help="lowest time value"
    )
    parser.add_argument(
        "-th", "--time_high", default=460, type=float, help="highest time value"
    )
    # Training options
    parser.add_argument("--save_every", type=int, default=5)

    parser.add_argument("--device", type=str, default="cuda")

    # train args

    parser.add_argument(
        "--batch_size", type=int, default=16, help="Batch size for training."
    )
    parser.add_argument(
        "--num_epochs_image", type=int, default=5, help="Number of epochs for training."
    )
    parser.add_argument(
        "--num_epochs_eeg", type=int, default=5, help="Number of epochs for training."
    )
    parser.add_argument(
        "--save_steps",
        default=5000,
        type=int,
        help="Number of steps between saving checkpoints.",
    )
    parser.add_argument(
        "--logging_steps", default=30, type=int, help="Number of steps between logging."
    )
    parser.add_argument(
        "--learning_rate",
        default=2e-5,
        type=float,
        help="The initial learning rate for Adam.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        default=4,
        type=int,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--optim",
        default="adamw_hf",
        type=str,
        help="Optimizer to use for training.",
    )
    parser.add_argument(
        "--weight_decay", default=0.001, type=float, help="Weight decay to apply."
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Whether to use 16-bit (mixed precision) training.",
    )
    parser.add_argument(
        "--bf16", action="store_true", help="Whether to use bfloat16 training."
    )
    parser.add_argument(
        "--max_grad_norm",
        default=0.3,
        type=float,
        help="Max gradient norm to clip gradients.",
    )
    parser.add_argument(
        "--max_steps",
        default=-1,
        type=int,
        help="If > 0: set total number of training steps to perform.",
    )
    parser.add_argument(
        "--warmup_ratio",
        default=0.3,
        type=float,
        help="Ratio of total steps to perform linear learning rate warmup.",
    )
    parser.add_argument(
        "--group_by_length",
        action="store_true",
        help="Whether to group samples of roughly the same length together.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        default="constant",
        type=str,
        help="Type of learning rate scheduler.",
    )
    parser.add_argument(
        "--report_to",
        default="tensorboard",
        type=str,
        help="Where to report training metrics.",
    )
    # Parse arguments
    args = parser.parse_args()
    return args


def get_args_for_llm_inference():
    # Define options
    parser = argparse.ArgumentParser(description="Template")

    # Dataset options

    ### BLOCK DESIGN ###
    # Data
    parser.add_argument(
        "--eeg_dataset", default=None, help="EEG dataset path"
    )  # 5-95Hz
    parser.add_argument("--image_dir", default=None, help="ImageNet dataset path")
    # Splits
    parser.add_argument(
        "--splits_path", default=None, help="splits path"
    )  # All subjects
    ### BLOCK DESIGN ###
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Directory to load the model checkpoints",
    )
    
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument(
        "-sn", "--split_num", default=0, type=int, help="split number"
    )  # leave this always to zero.

    # Subject selecting
    parser.add_argument(
        "-sub",
        "--subject",
        default=0,
        type=int,
        help="choose a subject from 1 to 6, default is 0 (all subjects)",
    )

    # Time options: select from 20 to 460 samples from EEG data
    parser.add_argument(
        "-tl", "--time_low", default=20, type=float, help="lowest time value"
    )
    parser.add_argument(
        "-th", "--time_high", default=460, type=float, help="highest time value"
    )
    parser.add_argument(
        "--dest",
        type=str,
        required=True,
        help="Directory to save the model checkpoints and logs.",
    )

    # Parse arguments
    args = parser.parse_args()
    return args
