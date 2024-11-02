from aria.tokenizer import AbsTokenizer
from aria.data.midi import MidiDict

with open("/project/jonmay_231/spangher/Projects/music-form-structure-modeling/aria-extended-gen-no-ft/synth_data/samples_0/1_style.txt", 'r') as file:
    content = file.read()
tokenizer = AbsTokenizer()
tokens = list(content)
encoded_tokens = tokenizer.encode(tokens)
print("file len:", len(encoded_tokens))

_midi_dict = MidiDict.from_midi("/project/jonmay_231/spangher/Projects/music-form-structure-modeling/aria-extended-gen-no-ft/synth_data/samples_0/1_midi.mid")
seq = tokenizer.tokenize(_midi_dict)
print("unfiltered seq len:", len(seq))
pure_seq = []
for tok in seq:
    if tok[0] in ['piano', 'onset', 'dur']:
        pure_seq.extend(tokenizer.encode([tok]))
print("filtered seq len:", len(pure_seq))

# count starting at <S>
start_idx = seq.index('<S>')
rm_metadata = seq[start_idx:]
test_seq = []
for tok in rm_metadata:
    if tok[0] in ['piano', 'onset', 'dur']:
        test_seq.extend(tokenizer.encode([tok]))
print("filtered seq len starting at <S>:", len(test_seq))