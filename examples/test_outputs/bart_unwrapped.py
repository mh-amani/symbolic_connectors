from blocks.modules.discrete_bottlenecks.softmax import SoftmaxDiscreteBottleneck
from blocks.models.encdec.bart import UnwrappedMbart, Unwrappedbart
from blocks.modules.model_unwrapper.transformer_enc_dec_unwrapper import EncoderDecoderUnwrapper
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast
import torch

def UnwrappedMBartTest():
    model_enfr = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")
    model_fren = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")

    vector_model_enfr, en_encoder_weight, fr_decoder_weight, fr_linearhead_weight = EncoderDecoderUnwrapper(model_enfr)
    vector_model_fren = EncoderDecoderUnwrapper(model_fren)
    discretizer_en = SoftmaxDiscreteBottleneck(vector_model_enfr)
    discretizer_fr = SoftmaxDiscreteBottleneck(vector_model_fren)

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


def UnwrappedBartTest():
    # an example for the encoder-decoder MBART model:
    # get the models and the discretizers
    unwrapped_model = UnwrappedMbart()
    model = unwrapped_model['model']
    vector_model = unwrapped_model['vector_model']
    en_discretizer = unwrapped_model['discretizer_enc']
    fr_discretizer = unwrapped_model['discretizer_dec']
    
    from transformers import MBart50TokenizerFast
    tokenizer = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50-many-to-many-mmt", src_lang="en_XX", tgt_lang="fr_XX")

    # an example input and output sequences
    sequence_en_1 = "Everything not saved will be lost."
    sequence_en_2 = "one must imagine Sisyphus happy."
    sequence_fr_1= "Tout ce qui n'est pas sauvé sera perdu." # "il m'a entarté"
    sequence_fr_2 = "il faut imaginer Sisyphe heureux."
    en_batch = [sequence_en_1, sequence_en_2]
    fr_batch = [sequence_fr_1, sequence_fr_2]
    input_enfr = tokenizer(text=en_batch, return_tensors="pt", padding=True)
    output_enfr = tokenizer(text_target=fr_batch, return_tensors="pt", padding=True) # the text_target is not doing anything!
    input_ids_enfr, input_attention_mask_enfr = input_enfr['input_ids'], input_enfr['attention_mask']
    output_ids_enfr, output_attention_mask_enfr = output_enfr['input_ids'], output_enfr['attention_mask']

    # add <EOS> to the beginning of labels ONLY FOR MBART CAUSE IT'S SO WEIRD
    output_ids_enfr = torch.cat((torch.ones((output_ids_enfr.shape[0], 1), dtype=torch.long).to(output_ids_enfr.device)*tokenizer.eos_token_id, output_ids_enfr), dim=1)
    output_attention_mask_enfr = torch.cat((torch.ones((output_attention_mask_enfr.shape[0], 1), dtype=torch.long).to(output_attention_mask_enfr.device), output_attention_mask_enfr), dim=1)
    
    # forward pass of one model
    input_vector_embeddings = en_discretizer.encoder_embedding_from_id(input_ids_enfr)
    output_vector_embeddings = fr_discretizer.decoder_embedding_from_id(output_ids_enfr) 
    output_vector_model = vector_model.forward(inputs_embeds=input_vector_embeddings, decoder_inputs_embeds=output_vector_embeddings,
                                            attention_mask=input_attention_mask_enfr, decoder_attention_mask=output_attention_mask_enfr,
                                            return_dict=True, output_hidden_states=True)
    discretized_output = fr_discretizer(output_vector_model['last_hidden_state'])
    # print the output of the discretizer discretized_output['id'], decoded with the tokenizer
    print('decoded output decomposed model:', tokenizer.batch_decode(discretized_output['id'], skip_special_tokens=False))
   
    # forward pass of the model without the discretizer (for comparison)
    # output_model = model_enfr(**input_enfr, labels=labels_enfr, return_dict=True, output_hidden_states=True)
    output_model = model(input_ids_enfr, attention_mask=input_attention_mask_enfr, 
                    decoder_input_ids=output_ids_enfr, decoder_attention_mask=output_attention_mask_enfr, return_dict=True)
    print('decoded output original model:', tokenizer.batch_decode(output_model.logits.argmax(dim=-1), skip_special_tokens=False))


if __name__ == '__main__':
    UnwrappedMBartTest()
    UnwrappedBartTest()