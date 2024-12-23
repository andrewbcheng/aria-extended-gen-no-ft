from aria.tokenizer import AbsTokenizer
from aria.data.midi import MidiDict

for j in range(1672, 1680):
    for i in range(1, 11):
        with open(f"/project/jonmay_231/spangher/Projects/music-form-structure-modeling/aria-extended-gen-no-ft/synth_data/samples_{j}/{i}_style.txt", 'r') as file:
            content = file.read()
        tokenizer = AbsTokenizer()
        tokens = list(content)
        #encoded_tokens = tokenizer.encode(tokens)

        _midi_dict = MidiDict.from_midi(f"/project/jonmay_231/spangher/Projects/music-form-structure-modeling/aria-extended-gen-no-ft/synth_data/samples_{j}/{i}_midi.mid")
        seq = tokenizer.tokenize(_midi_dict)
        tokens_in_midi_dict = 0
        pure_seq = []
        for tok in seq:
            if tok[0] in ['piano', 'onset', 'dur']:
                #pure_seq.extend(tokenizer.encode([tok]))
                tokens_in_midi_dict += 1

        # count starting at <S>
        #start_idx = seq.index('<S>')
        #rm_metadata = seq[start_idx:]
        #test_seq = []
        #for tok in rm_metadata:
        #    if tok[0] in ['piano', 'onset', 'dur']:
        #        test_seq.extend(tokenizer.encode([tok]))

        print("samples", j, "var", i, "tokens from midi dict:", tokens_in_midi_dict, "tokens from .txt:", len(tokens))