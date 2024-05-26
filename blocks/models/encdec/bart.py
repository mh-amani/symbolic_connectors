from blocks.modules.model_unwrapper.transformer_enc_dec_unwrapper import EncoderDecoderUnwrapper
from blocks.modules.discrete_bottlenecks.softmax import SoftmaxDiscreteBottleneck
from transformers import BartForConditionalGeneration

########################################################################################################################
# unwrapped models

def Unwrappedbart(bart_config):
    model = BartForConditionalGeneration(bart_config)
    vector_model, encoder_embedding, decoder_embedding, linear_head = EncoderDecoderUnwrapper(model).values()
    vocab_size = vector_model.config.vocab_size 
    embed_dim = vector_model.config.d_model
    discretizer_enc = SoftmaxDiscreteBottleneck({'dimensions': {'decoder_embedding_dim': embed_dim, 'vocab_size': vocab_size, 'encoder_embedding_dim': embed_dim, 'unembedding_dim': vocab_size}, 
                                'quantize_vector': True, 'temperature': 1.0,
                                'encoder_embedding_trainable': False, 'decoder_embedding_trainable': False, 'linear_head_trainable': False, 
                                'encoder_embedding': encoder_embedding, 'decoder_embedding': decoder_embedding, 
                                'linear_head': linear_head,})
    discretizer_dec = SoftmaxDiscreteBottleneck({'dimensions': {'decoder_embedding_dim': embed_dim, 'vocab_size': vocab_size, 'encoder_embedding_dim': embed_dim, 'unembedding_dim': vocab_size},
                                'quantize_vector': True, 'temperature': 1.0,
                                'encoder_embedding_trainable': False, 'decoder_embedding_trainable': False, 'linear_head_trainable': False,
                                'encoder_embedding': encoder_embedding, 'decoder_embedding': decoder_embedding,
                                'linear_head': linear_head,})
    return {
        'model': model, 'vector_model': vector_model,
        'discretizer_enc': discretizer_enc, 'discretizer_dec': discretizer_dec,}


def UnwrappedMbart(model=None, tokenizer=None, config=None):
    """
    Unwraps the MBART model to get the encoder and decoder weights.
    Returns:
    """
    from transformers import MBartForConditionalGeneration
    model = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")

    vector_model, encoder_embedding, decoder_embedding, linear_head = EncoderDecoderUnwrapper(model).values()
    vocab_size = vector_model.config.vocab_size
    embed_dim = vector_model.config.d_model

    discretizer_enc = SoftmaxDiscreteBottleneck({'dimensions': {'decoder_embedding_dim': embed_dim, 'vocab_size': vocab_size, 'encoder_embedding_dim': embed_dim, 'unembedding_dim': vocab_size},
                                'quantize_vector': True, 'temperature': 1.0,
                                'encoder_embedding_trainable': False, 'decoder_embedding_trainable': False, 'linear_head_trainable': False,
                                'encoder_embedding': encoder_embedding, 'decoder_embedding': decoder_embedding,
                                'linear_head': linear_head,})
    discretizer_dec = SoftmaxDiscreteBottleneck({'dimensions': {'decoder_embedding_dim': embed_dim, 'vocab_size': vocab_size, 'encoder_embedding_dim': embed_dim, 'unembedding_dim': vocab_size},
                                'quantize_vector': True, 'temperature': 1.0,
                                'encoder_embedding_trainable': False, 'decoder_embedding_trainable': False, 'linear_head_trainable': False,
                                'encoder_embedding': encoder_embedding, 'decoder_embedding': decoder_embedding,
                                'linear_head': linear_head,})
    
    return {
        'model': model, 'vector_model': vector_model,
        'discretizer_enc': discretizer_enc, 'discretizer_dec': discretizer_dec,}



########################################################################################################################
# unwrapped models
