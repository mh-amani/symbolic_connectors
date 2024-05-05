# from blocks.modules.discrete_bottleneck.softmax import SoftmaxDiscreteBottleneck
# from blocks.modules.blocks_connector import BlocksConnector


from transformers import MBartForConditionalGeneration, MBart50TokenizerFast

model_enfr = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")
model_fren = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")
tokenizer_enfr = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50-many-to-many-mmt", src_lang="en_XX", tgt_lang="fr_XX")
tokenizer_fren = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50-many-to-many-mmt", src_lang="fr_XX", tgt_lang="en_XX")

sequence_en_1 = "Everything not saved will be lost." # "he threw a pie at me"
sequence_en_2 = "one must imagine Sisyphus happy." # 
sequence_fr_1= "Tout ce qui n'est pas sauvé sera perdu." # "il m'a entarté"
sequence_fr_2 = "il faut imaginer Sisyphe heureux."

en_batch = [sequence_en_1, sequence_en_2]
fr_batch = [sequence_fr_1, sequence_fr_2]

# forward_pass English to French
input_enfr = tokenizer_enfr(text=en_batch, return_tensors="pt", padding=True)
labels_enfr = tokenizer_enfr(text_target=fr_batch, return_tensors="pt", padding=True).input_ids
# with tokenizer_enfr.as_target_tokenizer():
#     labels_enfr = tokenizer_enfr(fr_batch, return_tensors="pt", padding=True).input_ids

output_enfr = model_enfr(**input_enfr, labels=labels_enfr) # forward pass
print(output_enfr.loss) # => tensor(1.2345, grad_fn=<NllLossBackward>)
# printing predictions
print('input:' + sequence_en_1 + '\n' + 'output:' + tokenizer_enfr.decode(output_enfr.logits.argmax(dim=-1)[0])) # => "Bienvenue à NYC"
print('input:' + sequence_en_2 + '\n' + 'output:' + tokenizer_enfr.decode(output_enfr.logits.argmax(dim=-1)[1])) # => "HuggingFace est une entreprise", funny point: with this prompt it's most likely to predict in pashtu ()

# forward_pass French to English
input_label_fren = tokenizer_fren(text=fr_batch, text_target=en_batch, return_tensors="pt", padding=True)

output_fren = model_fren(**input_label_fren) # forward pass
print(output_fren.loss) # => tensor(1.2345, grad_fn=<NllLossBackward>)
# printing predictions
print('input:' + sequence_fr_1 + '\n' + 'output:' + tokenizer_fren.decode(output_fren.logits.argmax(dim=-1)[0])) # => "Welcome to NYC"
print('input:' + sequence_fr_2 + '\n' + 'output:' + tokenizer_fren.decode(output_fren.logits.argmax(dim=-1)[1])) # => "HuggingFace is a company"