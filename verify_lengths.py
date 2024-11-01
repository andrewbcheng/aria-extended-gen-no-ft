from aria.tokenizer import AbsTokenizer
from aria.data.midi import MidiDict

with open("/aria/synth_data/samples_991/1_style.txt", 'r') as file:
    content = file.read()
tokenizer = AbsTokenizer()
tokens = list(content)
encoded_tokens = tokenizer.encode(tokens)
print(len(encoded_tokens))

_midi_dict = MidiDict.from_midi("/aria/synth_data/samples_991/samples_991/1_midi.mid")
seq = tokenizer.tokenize(_midi_dict)
pure_seq = []
for tok in seq:
    if tok[0] in ['piano', 'onset', 'dur']:
        pure_seq.extend(tokenizer.encode([tok]))
print(len(pure_seq))