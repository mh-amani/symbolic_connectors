import torch 
from typing import Optional
from transformers import PreTrainedModel, EncoderDecoderModel
from transformers import MBartForConditionalGeneration, BartForConditionalGeneration
from blocks.modules.discrete_bottleneck.softmax import SoftmaxDiscreteBottleneck


def EncoderDecoderUnwrapper(enc_dec_model):
    """
    Unwraps the encoder-decoder model to get the encoder and decoder weights.
    Args:
        enc_dec_model: The encoder-decoder model.
    Returns:
        vector_model: The encoder-decoder model without embedding and head, pure transfomer.
        encoder_embedding_weight: The encoder weights.
        decoder_embedding_weight: The decoder weights.
        linearhead_weight: The linear head weights.
    """
    # Get the encoder and decoder weights
    encoder_embedding_weight = enc_dec_model.get_encoder().embed_tokens.weight.clone()
    encoder_embedding = torch.nn.Embedding(encoder_embedding_weight.shape[0], encoder_embedding_weight.shape[1])
    encoder_embedding.weight.data = encoder_embedding_weight
    try:
        encoder_embedding.weight.data = enc_dec_model.model.encoder.embed_scale * encoder_embedding.weight.data
    except:
        pass

    decoder_embedding_weight = enc_dec_model.get_decoder().embed_tokens.weight.clone()
    decoder_embedding = torch.nn.Embedding(decoder_embedding_weight.shape[0], decoder_embedding_weight.shape[1])
    decoder_embedding.weight.data = decoder_embedding_weight
    try:
        decoder_embedding.weight.data = enc_dec_model.model.decoder.embed_scale * decoder_embedding.weight.data
    except:
        pass

    linear_head_weight = enc_dec_model.lm_head.weight.clone()
    linear_head = torch.nn.Linear(linear_head_weight.shape[1], linear_head_weight.shape[0])
    linear_head.weight.data = linear_head_weight
    # linearhead_bias = enc_dec_model.lm_head.bias
    # linearhead_final_logit_bias = enc_dec_model.final_logits_bias
    # linear_head = enc_dec_model.get_output_embeddings()

    vector_model = enc_dec_model.model
    return {'vector_model': vector_model, 'encoder_embedding': encoder_embedding, 
        'decoder_embedding': decoder_embedding, 'linear_head': linear_head} 

########################################################################################################################
# Example to use the function vector_model_enfr, en_encoder_weight, fr_decoder_weight, fr_linearhead_weight = EncoderDecoderUnwrapper(model_enfr)


def Unwrappedbart(config):
    model = BartForConditionalGeneration(config)
    df = EncoderDecoderUnwrapper(model)
    vector_model = df['vector_model']
    encoder_embedding = df['encoder_embedding']
    decoder_embedding = df['decoder_embedding']
    linear_head= df['linear_head']
    return vector_model, encoder_embedding, decoder_embedding, linear_head

def UnwrappedMbart(model=None, tokenizer=None, config=None):
    """
    Unwraps the MBART model to get the encoder and decoder weights.
    Returns:
        model_enfr: The English to French model.
        model_fren: The French to English model.
        vector_model_enfr: The English to French model without embedding and head, pure transfomer.
        vector_model_fren: The French to English model without embedding and head, pure transfomer.
        discretizer_en: The English to French discretizer.
        discretizer_fr: The French to English discretizer.
        tokenizer_enfr: The English to French tokenizer.
        tokenizer_fren: The French to English tokenizer.
    """
    from transformers import MBartForConditionalGeneration
    model = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")

    vector_model, encoder_embedding, decoder_embedding, linear_head = EncoderDecoderUnwrapper(model).values()
    
    discretizer_enc = SoftmaxDiscreteBottleneck({'dimensions': {'decoder_embedding_dim': 1024, 'vocab_size': 250054, 'encoder_embedding_dim': 1024, 'unembedding_dim': 250054}, 
                                'quantize_vector': True, 'temperature': 1.0,
                                'encoder_embedding_trainable': False, 'decoder_embedding_trainable': False, 'linear_head_trainable': False, 
                                'encoder_embedding': encoder_embedding, 'decoder_embedding': decoder_embedding, 
                                'linear_head': linear_head,})
    discretizer_dec = SoftmaxDiscreteBottleneck({'dimensions': {'decoder_embedding_dim': 1024, 'vocab_size': 250054, 'encoder_embedding_dim': 1024, 'unembedding_dim': 250054}, 
                                'quantize_vector': True, 'temperature': 1.0,
                                'encoder_embedding_trainable': False, 'decoder_embedding_trainable': False, 'linear_head_trainable': False, 
                                'encoder_embedding': encoder_embedding, 'decoder_embedding': decoder_embedding,
                                'linear_head': linear_head,})
    return {
        'model': model, 'vector_model': vector_model,
        'discretizer_enc': discretizer_enc, 'discretizer_dec': discretizer_dec,}





def main() -> Optional[float]:
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


if __name__ == "__main__":
    main()

########################################################################################################################
# code snippets and other useful stuff for debugging and checking stuff

# model.model.encoder.embed_tokens(input_enfr['input_ids']) == input_vector_embeddings
# model.model.encoder.embed_positions(input_enfr['input_ids'])
# model.model.decoder.embed_tokens(labels_enfr['input_ids'])*32 - output_vector_embeddings

# # print the weights and shapes
    # print('-'*50)
    # print('vector model:', vector_model)  
    # print('encoder weight shape:', encoder_embedding_weight.shape)
    # print('decoder weight shape:', decoder_embedding_weight.shape)
    # print('linear head weight shape:', linearhead_weight.shape)
    # print('linear head weight:', linearhead_weight)
    # print('-'*50)


############################################################
# hard to make the forwardpass work, the lm_head is not removed properly

# def create_headless_transformer_model(base_model_class: PreTrainedModel, config):
#     """
#     Creates a modified version of a given transformer model class, without embedding and head layers.
    
#     Args:
#         base_model_class (PreTrainedModel): A class of transformer model from Hugging Face's transformers.
#         config: The configuration object for the model.

#     Returns:
#         A modified model class with no embedding and head layers.
#     """

#     class HeadlessTransformerModel(base_model_class):
#         def __init__(self, *args, **kwargs):
#             super().__init__(*args, **kwargs)
#             # Removing embedding layers and head might depend on the specific model architecture
#             self.model.encoder.embed_tokens = None
#             self.model.decoder.embed_tokens = None
#             self.lm_head = None
        
#         def forward(self, *args, **kwargs):
#             """
#             The forward pass will need to be adjusted depending on what the default model expects
#             and what is now missing (like embeddings and heads).
#             """
#             # Typical use-case would involve calling original model's forward method,
#             # But you must handle the lack of embeddings and heads appropriately.
#             outputs = super().forward(*args, **kwargs)
#             return outputs
    
#     # Return a new class instance with removed components
#     return HeadlessTransformerModel(config)

############################################################
# # vector_model.lm_head = lambda x: x
    # Replace the lm_head with an Identity module
    # vector_model.lm_head = torch.nn.Identity()

    # things that fail to work, especially the forward pass breaks...
    # Remove the embedding and head from the model
    # vector_model = enc_dec_model
    # vector_model.get_encoder().embed_tokens = None
    # vector_model.get_decoder().embed_tokens = None
    # vector_model.lm_head = None
    # vector_model = create_headless_transformer_model(enc_dec_model.__class__, enc_dec_model.config)

    # encoder_model = enc_dec_model.get_encoder()
    # decoder_model = enc_dec_model.get_decoder()
    # vector_model = torch.nn.ModuleDict({'encoder': encoder_model, 'decoder': decoder_model})
    # vector_model = EncoderDecoderModel(encoder=encoder_model, decoder=decoder_model)
    # vector_model.get_encoder().embed_tokens = None
    # vector_model.get_decoder().embed_tokens = None


############################################################
    # an example for the encoder-decoder MBART model:
    # Initialize the model
    # from transformers import MBartForConditionalGeneration, MBart50TokenizerFast
    # model = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")
    # # tokenizing the input and output sequences (teacher forced generation)
    # tokenizer_enfr = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50-many-to-many-mmt", src_lang="en_XX", tgt_lang="fr_XX")
    
    # # Unwrap the model to get the encoder and decoder weights, so you can initialize discretizers
    # unwrapped_model = EncoderDecoderUnwrapper(model)
    # vector_model, encoder_embedding_weight, decoder_embedding_weight, linearhead_weight, linearhead_bias = unwrapped_model.values()
    # # initialize the discretizer and connect the two models via the blocks module
    # discretizer_config = {'dimensions': {'decoder_embedding_dim': 1024, 'vocab_size': tokenizer_enfr.vocab_size, 'encoder_embedding_dim': 1024, 'unembedding_dim': tokenizer_enfr.vocab_size}, 
    #                             'quantize_vector': True, 'temperature': 1.0,
    #                             'encoder_embedding_trainable': False, 'decoder_embedding_trainable': False, 'linear_head_trainable': False, 
    #                             'encoder_embedding_weight': encoder_embedding_weight, 'decoder_embedding_weight': decoder_embedding_weight, 
    #                             'linear_head_weight': linearhead_weight, 'linear_head_bias': linearhead_bias}
                        
    # en_discretizer = SoftmaxDiscreteBottleneck(discretizer_config)
    # fr_discretizer = SoftmaxDiscreteBottleneck(discretizer_config)