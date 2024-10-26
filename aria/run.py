#!/usr/bin/env python3

import argparse
import os
import re
import sys
import copy


def _parse_sample_args():
    argp = argparse.ArgumentParser(prog="aria sample")
    argp.add_argument("-m", help="name of model config file")
    argp.add_argument("-c", help="path to model checkpoint")
    argp.add_argument( # paths to all the midi files we will use 
        "-p",
        help="paths to all midi files",
        type=str,
        nargs="+",
        )
    argp.add_argument(
        "-form",
        help="form of the generation",
        type=str,
    )
    argp.add_argument(
        "-pt", help="sample using the pretrained model", action="store_true"
    )
    argp.add_argument(
        "-temp",
        help="change temp value",
        type=float,
        required=False,
        default=0.95,
    )
    argp.add_argument(
        "-top_p",
        help="change top_p value",
        type=float,
        required=False,
        default=0.95,
    )
    argp.add_argument(
        "-metadata",
        nargs=2,
        metavar=("KEY", "VALUE"),
        action="append",
        help="manually add metadata key-value pair when sampling",
    )
    argp.add_argument(
        "-var",
        help="number of variations",
        type=int,
        default=1,
    )
    argp.add_argument(
        "-trunc",
        help="length (in seconds) of the prompt",
        type=int,
        default=20,
    )
    argp.add_argument("-e", action="store_true", help="enable force end")
    argp.add_argument("-l", type=int, help="generation length", default=1024)
    argp.add_argument("-noise", action="store_true", help="add noise to prompt")
    argp.add_argument("-compile", action="store_true", help="compile cudagraph")

    return argp.parse_args(sys.argv[2:])


def _get_model_name(name: str | None, state: dict):
    if name is not None:
        return name

    print("Model name is not provided. Trying to infer from checkpoint...")
    _defaults = {
        16: "small",
        32: "medium",
        64: "large",
    }
    try:
        pattern = re.compile(r"encode_layers\.(\d+)\.")
        layer_keys = [pattern.search(k) for k in state.keys()]
        layer_keys = set(p.group(1) for p in layer_keys if p is not None)
        for i in range(len(layer_keys)):
            assert str(i) in layer_keys

        if len(layer_keys) in _defaults:
            print(f"Selecting model name: {_defaults[len(layer_keys)]}")
            return _defaults[len(layer_keys)]
        assert False
    except:
        raise ValueError("Model name is not provided and cannot be inferred.")


# TODO: Add support for sampling from the pretrained model
def sample(args):
    """Entrypoint for sampling"""

    from torch.cuda import is_available as cuda_is_available
    from aria.inference import TransformerLM
    from aria.model import ModelConfig
    from aria.config import load_model_config, load_config
    from aria.tokenizer import AbsTokenizer, SeparatedAbsTokenizer
    from aria.sample import greedy_sample, get_pt_prompt, get_inst_prompt
    from aria.data.midi import MidiDict
    from aria.data.datasets import _noise_midi_dict
    from aria.utils import midi_to_audio, _load_weight

    if not cuda_is_available():
        raise Exception("CUDA device is not available.")

    model_state = _load_weight(args.c, "cuda")
    model_state = {
        k: v for k, v in model_state.items() if "rotary_emb" not in k
    }

    manual_metadata = {k: v for k, v in args.metadata} if args.metadata else {}
    valid_metadata = load_config()["data"]["metadata"]["manual"]
    for k, v in manual_metadata.copy().items():
        assert k in valid_metadata.keys(), f"{manual_metadata} is invalid"
        if v not in valid_metadata[k]:
            print(f"Ignoring invalid manual metadata: {k}")
            print(f"Please choose from {valid_metadata[k]}")
            del manual_metadata[k]

    num_variations = args.var
    truncate_len = args.trunc
    force_end = args.e
    model_name = args.m

    if args.pt == True:
        tokenizer = AbsTokenizer(return_tensors=True)
    else:
        tokenizer = SeparatedAbsTokenizer(return_tensors=True)

    model_config = ModelConfig(**load_model_config(model_name))
    model_config.set_vocab_size(tokenizer.vocab_size)
    model_config.grad_checkpoint = False
    model = TransformerLM(model_config).cuda()

    try:
        model.load_state_dict(model_state)
    except Exception as e:
        print(
            "Failed to load model_state. This is likely due to an incompatibility "
            "between the checkpoint file (-c) and model name/config (-m)."
        )
        if args.pt:
            print(
                "When using the -pt flag make sure you provide a checkpoint for "
                "the pretrained model."
            )
        else:
            print(
                "When not using the -pt flag make sure you provide a checkpoint "
                " for the instuct-finetuned (inst) model."
            )

        raise e

    assert args.l > 0, "Generation length must be positive."
    max_new_tokens = args.l

    # Load and format prompts and metadata
    midi_dict_paths = args.p # [tests/test_data/1.mid, tests/test_data/2.mid]
    form = args.form # 'ABA'
    
    midi_dicts = {}
    path_idx = 0
    for section in form:
        if section not in midi_dicts.keys():
            midi_dict = MidiDict.from_midi(mid_path=midi_dict_paths[path_idx])
            
            for k, v in manual_metadata.items():
                midi_dict.metadata[k] = v
                
            print(f"Extracted metadata: {midi_dict.metadata}")
            print(
                f"Instruments: {set([MidiDict.get_program_to_instrument()[msg['data']] for msg in midi_dict.instrument_msgs])}"
            )
            
            midi_dicts[section] = midi_dict
            path_idx += 1
        
    print(f"Number of MidiDicts: {len(midi_dicts.keys())}")
    print(f"Form to follow: {form}")    
    
    final_results = [] # [[A1_tokens, ...], [B1_tokens, ...], [A1_tokens, ...]]  
    generated = {} # {A : [A1_tokens, ...], B : [B1_tokens, ...]}
    
    for section in form:
        if args.pt:
            if args.noise:
                print("Noising not supported with pretrained model")
                
            prompt_seq = get_pt_prompt(
                tokenizer=tokenizer,
                midi_dict=midi_dicts[section],
                truncate_len=truncate_len,
            )
        else:
            prompt_seq = get_inst_prompt(
                tokenizer=tokenizer,
                midi_dict=midi_dicts[section],
                truncate_len=truncate_len,
                noise=args.noise,
            )
            
        prompts = [prompt_seq for _ in range(num_variations)]
        if len(prompt_seq) + args.l > model_config.max_seq_len:
            print(
                "WARNING: Required context exceeds max_seq_len supported by model"
            )
        print(prompt_seq)
        
        # select which generation to use
        if section in generated:
            results = copy.deepcopy(generated[section])
        else:
            results = greedy_sample(
                model=model,
                tokenizer=tokenizer,
                prompts=prompts,
                max_new_tokens=max_new_tokens,
                force_end=True,
                temperature=args.temp,
                top_p=args.top_p,
                compile=args.compile,
            )
            generated[section] = results # [A1_tokens, ...]
        final_results.append(results) # [[A1_tokens, ...], [B1_tokens, ...], [A1_tokens, ...]]
    
    token_labels = [[] for _ in range(num_variations)] # [[# A1 + # B1 + # A1], ...]
    for idx_section, section in enumerate(final_results): # [A1_tokens, ...]
        if idx_section == 0: # grab original ending ticks
            ending_ticks = [0 for _ in range(num_variations)] # [A1_ending_ticks, ...]
            final_midi_dicts = [] # [A1_mididict + B1_mididict (tick-modified) + A1_mididict (tick-modified), ...]
            
            for idx_seq, seq in enumerate(section):
                res_midi_dict = tokenizer.detokenize(seq) # A1_mididict
                final_midi_dicts.append(res_midi_dict) # [A1_mididict, ...]
                ending_ticks[idx_seq] = res_midi_dict.note_msgs[-1]["data"]["end"] + 2000
        
        else: # modifying current and provide new ending ticks for next iteration
            for idx_seq, seq in enumerate(section):
                res_midi_dict = tokenizer.detokenize(seq) # B1_mididict
                
                for msg in res_midi_dict.note_msgs: # [A1_mididict + B1_mididict (tick-modified), ...]
                    adjusted_note_msg = copy.deepcopy(msg)
                    adjusted_note_msg["tick"] += ending_ticks[idx_seq]
                    adjusted_note_msg["data"]["start"] += ending_ticks[idx_seq]
                    adjusted_note_msg["data"]["end"] += ending_ticks[idx_seq]
                    
                    final_midi_dicts[idx_seq].note_msgs.append(adjusted_note_msg) 
            
            for idx_seq, seq in enumerate(final_midi_dicts): # ending tick of last note of B1_mididict (tick-modified)
                ending_ticks[idx_seq] = seq.note_msgs[-1]["data"]["end"] + 2000
                
        # labels
        for idx_seq, seq in enumerate(section):
            start_idx = seq.index('<S>')
            rm_metadata = seq[start_idx:]
            #note_tokens = [tok for tok in rm_metadata if type(tok) is tuple]
            note_tokens = [tok for tok in rm_metadata if tok[0] in ['piano', 'onset', 'dur']] #count tokens correctly, ignore INST tokens
            for _ in range(len(note_tokens)):
                token_labels[idx_seq].append(form[idx_section])

    samples_dir = os.path.join(os.path.dirname(__file__), "..", "synth_data/samples_0")
    if os.path.isdir(samples_dir) is False:
        os.mkdir(samples_dir)
    else:
        sample_num = 0
        while os.path.isdir(samples_dir):
            sample_num += 1
            samples_dir = os.path.join(os.path.dirname(__file__), "..", "synth_data/samples_") + str(sample_num)
        os.mkdir(samples_dir)
            
    for idx, token_label in enumerate(token_labels): # [A1 + # B1 + # A1] 
        output_file = os.path.join(samples_dir, f"{idx + 1}_style.txt")
        with open(output_file, 'w') as file:
            for tok in token_label:
                file.write(tok)
            
    print(f"Labels saved to {samples_dir}")

    for idx, combined_midi_dict in enumerate(final_midi_dicts):
        res_midi = combined_midi_dict.to_midi()
        res_midi.save(f"{samples_dir}/{idx + 1}_midi.mid")

    print(f"Results saved to {samples_dir}")


def _parse_midi_dataset_args():
    argp = argparse.ArgumentParser(prog="aria midi-dataset")
    argp.add_argument("dir", help="directory containing midi files")
    argp.add_argument("save_path", help="path to save dataset")
    argp.add_argument("-r", action="store_true", help="recursively search dirs")
    argp.add_argument(
        "-s", action="store_true", help="shuffle dataset", default=False
    )
    argp.add_argument(
        "-metadata",
        nargs=2,
        metavar=("KEY", "VALUE"),
        action="append",
        help="manually add metadata key-value pair when building dataset",
    )
    argp.add_argument(
        "-split", type=float, help="create train/val split", required=False
    )

    return argp.parse_args(sys.argv[2:])


def build_midi_dataset(args):
    """Entrypoint for building MidiDatasets from a directory"""
    from aria.data.datasets import MidiDataset

    assert args.dir, "build directory must be provided"
    manual_metadata = {k: v for k, v in args.metadata} if args.metadata else {}
    MidiDataset.build_to_file(
        dir=args.dir,
        save_path=args.save_path,
        recur=args.r,
        overwrite=True,
        manual_metadata=manual_metadata,
        shuffle=args.s,
    )

    if args.split:
        assert 0.0 < args.split < 1.0, "Invalid range given for -split"
        MidiDataset.split_from_file(
            load_path=args.save_path,
            train_val_ratio=args.split,
            repeatable=True,
        )


def _parse_pretrain_dataset_args():
    argp = argparse.ArgumentParser(prog="aria pretrain-dataset")
    argp.add_argument("-load_path", help="path midi_dict dataset")
    argp.add_argument("-save_dir", help="path to save dataset")
    argp.add_argument(
        "-tokenizer_name", help="tokenizer name", choices=["abs", "rel"]
    )
    argp.add_argument("-l", help="max sequence length", type=int, default=4096)
    argp.add_argument("-e", help="num epochs", type=int, default=1)

    return argp.parse_args(sys.argv[2:])


def build_pretraining_dataset(args):
    from aria.tokenizer import AbsTokenizer, RelTokenizer
    from aria.data.datasets import PretrainingDataset

    if args.tokenizer_name == "abs":
        tokenizer = AbsTokenizer()
    elif args.tokenizer_name == "rel":
        tokenizer = RelTokenizer()

    dataset = PretrainingDataset.build(
        tokenizer=tokenizer,
        save_dir=args.save_dir,
        max_seq_len=args.l,
        num_epochs=args.e,
        midi_dataset_path=args.load_path,
    )


def _parse_finetune_dataset_args():
    argp = argparse.ArgumentParser(prog="aria finetune-dataset")
    argp.add_argument(
        "-clean_load_path",
        help="path to the clean midi_dict dataset",
    )
    argp.add_argument(
        "-noisy_load_paths",
        nargs="+",
        help="one or more paths to noisy midi_dict datasets",
    )
    argp.add_argument("-save_dir", help="path to save dataset")
    argp.add_argument("-l", help="max sequence length", type=int, default=4096)
    argp.add_argument("-e", help="num epochs", type=int, default=1)

    return argp.parse_args(sys.argv[2:])


def build_finetune_dataset(args):
    from aria.tokenizer import SeparatedAbsTokenizer
    from aria.data.datasets import FinetuningDataset

    tokenizer = SeparatedAbsTokenizer()
    dataset = FinetuningDataset.build(
        tokenizer=tokenizer,
        save_dir=args.save_dir,
        max_seq_len=args.l,
        num_epochs=args.e,
        clean_dataset_path=args.clean_load_path,
        noisy_dataset_paths=args.noisy_load_paths,
    )


def main():
    # Nested argparse inspired by - https://shorturl.at/kuKW0
    parser = argparse.ArgumentParser(usage="aria <command> [<args>]")
    parser.add_argument(
        "command",
        help="command to run",
        choices=(
            "sample",
            "midi-dataset",
            "pretrain-dataset",
            "finetune-dataset",
        ),
    )

    # parse_args defaults to [1:] for args, but you need to
    # exclude the rest of the args too, or validation will fail
    args = parser.parse_args(sys.argv[1:2])

    if not hasattr(args, "command"):
        parser.print_help()
        print("Unrecognized command")
        exit(1)
    elif args.command == "sample":
        sample(args=_parse_sample_args())
    elif args.command == "midi-dataset":
        build_midi_dataset(args=_parse_midi_dataset_args())
    elif args.command == "pretrain-dataset":
        build_pretraining_dataset(args=_parse_pretrain_dataset_args())
    elif args.command == "finetune-dataset":
        build_finetune_dataset(args=_parse_finetune_dataset_args())
    else:
        print("Unrecognized command")
        parser.print_help()
        exit(1)


if __name__ == "__main__":
    main()
