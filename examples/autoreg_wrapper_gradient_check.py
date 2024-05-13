import torch
from transformers import BartForConditionalGeneration, BartConfig
from blocks.modules.auto_reg_wrapper import AutoRegWrapper
from blocks.unwrapped_models.enc_dec_unwrapper import EncoderDecoderUnwrapper
from blocks.modules.discrete_bottleneck.softmax import SoftmaxDiscreteBottleneck


def test_gradients_vector_model_functionality():
    device='cuda' if torch.cuda.is_available() else 'cpu'
    vocab_size = 4
    batch_size = 1
    seq_length = 4
    emb_dim = 8
    config_bart ={
        'vocab_size': vocab_size,
        'max_position_embeddings': seq_length,
        'encoder_layers': 1,
        'encoder_ffn_dim': emb_dim,
        'encoder_attention_heads': 1,
        'decoder_layers': 1,
        'decoder_ffn_dim': emb_dim,
        'decoder_attention_heads': 1,
        'd_model': emb_dim,
        'use_cache': True,
        'torch_dtype': 'float32'
    }
    discretizer_config = {
        device: device,
        'dimensions': {
            'encoder_embedding_dim': emb_dim,
            'decoder_embedding_dim': emb_dim,
            'vocab_size': vocab_size,
            'unembedding_dim': vocab_size
        },
        'encoder_embedding_trainable': True,
        'decoder_embedding_trainable': True,
        'linear_head_trainable': True,
        'quantize_vector': True,
        'temperature': 0.1 # Low temperature should make softmax near one-hot
    }

    # fix seed
    torch.manual_seed(45)

    model = BartForConditionalGeneration(BartConfig(**config_bart)).to(device)
    vector_model, encoder_embedding, decoder_embedding, linearhead = EncoderDecoderUnwrapper(model).values()
    discretizer = SoftmaxDiscreteBottleneck({**discretizer_config, 'encoder_embedding': encoder_embedding,
                                             'decoder_embedding': decoder_embedding, 'linear_head': linearhead,}).to(device)
    

    input_ids = torch.randint(0, vocab_size, (batch_size, seq_length)).to(device)
    output_ids = torch.randint(0, vocab_size, (batch_size, seq_length)).to(device)
    output_ids[:, 0] = 0

    # Forward pass
    input_embeds_0 = discretizer.encoder_embedding_from_id(input_ids)
    output_embeds_0 = discretizer.encoder_embedding_from_id(output_ids[:, 0:1])
    input_attention_mask_0 = torch.ones(output_embeds_0.shape[:2]).to(device)
    output_vector_model_1 = vector_model(inputs_embeds=input_embeds_0, # decoder_attention_mask=input_attention_mask_0,  #doesn't make a difference
                                         decoder_inputs_embeds=output_embeds_0,)['last_hidden_state']
    discrete_output_1 = discretizer(output_vector_model_1)
    discrete_output_1_scores = discrete_output_1['score']
    discrete_output_1_encoder_embeds = discrete_output_1['quantized_vector_encoder']
    discrete_output_1_decoder_embeds = discrete_output_1['quantized_vector_decoder']
    output_embeds_1 = torch.cat((output_embeds_0, discrete_output_1_decoder_embeds), dim=1)
    input_attention_mask_1 = torch.ones(output_embeds_1.shape[:2]).to(device)
    output_vector_model_2 = vector_model(inputs_embeds=input_embeds_0, # decoder_attention_mask=input_attention_mask_1,  #doesn't make a difference
                                         decoder_inputs_embeds=output_embeds_1, )['last_hidden_state']
    discrete_output_2 = discretizer(output_vector_model_2)
    discrete_output_2_scores = discrete_output_2['score']

    # torch.nn.functional.cross_entropy(discrete_output_2_scores[0, 1:], output_ids[:, 1].view(-1))
    loss = torch.nn.functional.nll_loss(torch.log(discrete_output_2_scores[0, 1:]), output_ids[:, 1].view(-1))
    
    # retain_grad
    input_embeds_0.retain_grad()
    discrete_output_1_scores.retain_grad()
    discrete_output_1_encoder_embeds.retain_grad()
    discrete_output_1_decoder_embeds.retain_grad()
    discrete_output_2_scores.retain_grad()

    loss.backward()
    print(f"Gradient of the loss w.r.t. the input encoder embedding: {input_embeds_0.grad}")
    print(f"Gradient of the loss w.r.t. the output1 encoder embedding: {discrete_output_1_encoder_embeds.grad}")
    print(f"Gradient of the loss w.r.t. the output1 decoder embedding: {discrete_output_1_decoder_embeds.grad}")
    print(f"Gradient of the loss w.r.t. the output1 probabilities: {discrete_output_1_scores.grad}")
    print(f"Gradient of the loss w.r.t. the output2 probabilities: {discrete_output_2_scores.grad}")
    print(f"gradient of loss w.r.t. the encoder embedding parameters: {discretizer.encoder_embedding.weight.grad}")
    print(f"gradient of loss w.r.t. the decoder embedding parameters: {discretizer.decoder_embedding.weight.grad}")
    print(f"gradient of loss w.r.t. the linear head parameters: {discretizer.linear_head.weight.grad}")
    print(f"Loss: {loss}")


def test_gradients_autoregged_wrapped_functionality():
    device='cuda' if torch.cuda.is_available() else 'cpu'
    vocab_size = 4
    batch_size = 1
    seq_length = 4
    emb_dim = 8
    config = {
    'device': device,
    'output_prepending_ids': torch.tensor([0]),
    'use_past_key_values': False, 'use_last_step_states': True,
    'max_lengths': {'input': 5, 'output': 5,},
    'control_token_ids': { 'input_pad_token_id': 0,
                            'output_eos_token_id': 3,
                            'output_pad_token_id': 0,
                            'output_unknown_token_id': 0,},
    'soft_average': {'p_eos_backward': True, 'p_eos_forward': False ,'word_embeds_with_scores_forward': True,}}
    config_bart ={
    'vocab_size': vocab_size,
    'max_position_embeddings': seq_length,
    'encoder_layers': 1,
    'encoder_ffn_dim': emb_dim,
    'encoder_attention_heads': 1,
    'decoder_layers': 1,
    'decoder_ffn_dim': emb_dim,
    'decoder_attention_heads': 1,
    'd_model': emb_dim,
    'use_cache': True,
    'torch_dtype': 'float32'
    }
    discretizer_config = {
        device: device,
        'dimensions': {
            'encoder_embedding_dim': emb_dim,
            'decoder_embedding_dim': emb_dim,
            'vocab_size': vocab_size,
            'unembedding_dim': vocab_size
        },
        'encoder_embedding_trainable': True,
        'decoder_embedding_trainable': True,
        'linear_head_trainable': True,
        'quantize_vector': True,
        'temperature': 1 # Low temperature should make softmax near one-hot
    }

    # fix seed
    torch.manual_seed(42)
    
    model = BartForConditionalGeneration(BartConfig(**config_bart)).to(device)
    vector_model, encoder_embedding, decoder_embedding, linear_head = EncoderDecoderUnwrapper(model).values()
    discretizer = SoftmaxDiscreteBottleneck({**discretizer_config, 'encoder_embedding': encoder_embedding,
                                            'decoder_embedding': decoder_embedding, 'linear_head': linear_head,}).to(device)
    
    enfr_autoreg_wrapped_model = AutoRegWrapper(vector_model, discretizer, discretizer,config).to(device)

    input_ids = torch.randint(0, vocab_size, (batch_size, seq_length)).to(device)
    output_ids = torch.randint(0, vocab_size, (batch_size, seq_length)).to(device)
    output_ids[:, 0] = 0

    # Forward pass
    output = enfr_autoreg_wrapped_model(input_ids, max_output_length=seq_length)
    scores = output['score']
    encoder_embeds = output['quantized_vector_encoder']
    decoder_embeds = output['quantized_vector_decoder']
    loss = torch.nn.functional.nll_loss(torch.log(scores[:, 1]), output_ids[:, 1].view(-1))

    # retain_grad
    encoder_embeds.retain_grad()
    decoder_embeds.retain_grad()
    scores.retain_grad()

    loss.backward()
    print(f"Gradient of the loss w.r.t. the input encoder embedding: {encoder_embeds.grad}")
    print(f"Gradient of the loss w.r.t. the input decoder embedding: {decoder_embeds.grad}")
    print(f"Gradient of the loss w.r.t. the output probabilities: {scores.grad}")
    print(f"gradient of loss w.r.t. the embedding parameters: {discretizer.encoder_embedding.weight.grad}")
    print(f"gradient of loss w.r.t. the linear head parameters: {discretizer.linear_head.weight.grad}")
    print(f"gradient of loss w.r.t. the decoder embedding parameters: {discretizer.decoder_embedding.weight.grad}")
    print(f"Loss: {loss}")



if __name__ == '__main__':
    test_gradients_vector_model_functionality()
    # test_gradients_autoregged_wrapped_functionality()   
