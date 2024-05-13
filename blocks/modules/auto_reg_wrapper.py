from typing import Any, Dict, Tuple
import torch
from torch.nn import ModuleDict, Module
from transformers import MBart50TokenizerFast  
from blocks.unwrapped_models.enc_dec_unwrapper import UnwrappedMbart


class AutoRegWrapper(Module):
    """
    a wrapper connecting two sequence models with discrete bottleneck layers
    """
    def __init__(
        self,
        vector_model,
        input_discretizer,
        output_discretizer,
        config,
    ) -> None:

        super().__init__()
        self.config = config
        self.model = vector_model
        self.input_discretizer = input_discretizer
        self.output_discretizer = output_discretizer
        self.max_lengths = self.config['max_lengths']
        self.control_token_ids = self.config['control_token_ids']
        self.soft_average = self.config['soft_average']
        device=self.config['device']

        self.output_discretizer.to(device)
        self.input_discretizer.to(device)

        self.pad_embed_enc = self.output_discretizer.encoder_embedding_from_id(torch.tensor(self.control_token_ids['output_pad_token_id']).to(self.output_discretizer.encoder_embedding.weight.device))
        self.pad_embed_dec = self.output_discretizer.decoder_embedding_from_id(torch.tensor(self.control_token_ids['output_pad_token_id']).to(self.output_discretizer.encoder_embedding.weight.device))
        self.onehot_score_pad = torch.nn.functional.one_hot(torch.tensor(self.control_token_ids['output_pad_token_id']), num_classes=self.output_discretizer.vocab_size).to(self.output_discretizer.encoder_embedding.weight.device).float()

        output_prepending_ids = self.config.get('output_prepending_ids', None)
        output_prepending_embeds_enc = self.config.get('output_prepending_embeds_enc', None)
        output_prepending_embeds_dec = self.config.get('output_prepending_embeds_dec', None)

        if output_prepending_ids is None and (output_prepending_embeds_enc is None or output_prepending_embeds_dec is None):
            raise ValueError("output_prepending_ids nor the embeddings are not provided")
        elif output_prepending_ids is None and (output_prepending_embeds_enc is not None and output_prepending_embeds_dec is not None):
            self.output_prepending_ids = self.control_token_ids['pad_token_id_y'] * torch.ones(output_prepending_embeds_dec.shape[:2], dtype=torch.long).to(self.onehot_score_pad.device)
            self.output_prepending_embeds_enc = output_prepending_embeds_enc.to(self.onehot_score_pad.device)
            self.output_prepending_embeds_dec = output_prepending_embeds_dec.to(self.onehot_score_pad.device)
        else:
            self.output_prepending_ids = output_prepending_ids.to(self.config['device'])
            self.output_prepending_embeds_enc = self.output_discretizer.encoder_embedding_from_id(self.output_prepending_ids).to(self.config['device'])
            self.output_prepending_embeds_dec = self.output_discretizer.decoder_embedding_from_id(self.output_prepending_ids).to(self.config['device'])


    def forward(self, input_ids: torch.Tensor=None, input_attention_mask: torch.Tensor=None, input_embeds_enc: torch.Tensor=None,
                output_ids: torch.Tensor=None,
                output_embeds_enc: torch.Tensor=None, output_embeds_dec: torch.Tensor=None, output_attention_mask: torch.Tensor=None,
                teacher_force_output: bool=False, max_output_length=None) -> Dict[str, torch.Tensor]:
        """Perform a forward pass through the models x -> y_1 y_2 ... y_n

        """
        # assert that only one of the embeds or ids is provided, and attention mask should be provided if embeds are provided

        if max_output_length is None:
            max_output_length = self.max_lengths['output']
        assert (input_ids is not None) != (input_embeds_enc is not None), "Either input_ids or input_embeds should be provided"
        assert (input_embeds_enc is not None and input_attention_mask is not None) or (input_embeds_enc is None and input_attention_mask is None), "input_embeds and input_attention_mask should be provided together or not at all"
        assert (output_ids is None)  or (output_embeds_enc is None), "Either output_ids or output_embeds or neither should be provided, but not both"
        assert (output_embeds_enc is not None and output_embeds_dec is not None and output_attention_mask is not None) or (output_embeds_enc is None and output_embeds_dec is None and output_attention_mask is None), "output_embeds and output_attention_mask should be provided together or not at all"

        if input_ids is not None:
            input_embeds_enc = self.input_discretizer.encoder_embedding_from_id(input_ids)
            input_attention_mask = torch.logical_not(torch.eq(input_ids, self.control_token_ids['input_pad_token_id']))
        
        if output_ids is not None:
            # setting the output encoder and decoder embeddings and attention mask, starting generation from these embeddings
            output_embeds_enc = self.output_discretizer.encoder_embedding_from_id(output_ids)
            output_embeds_dec = self.output_discretizer.decoder_embedding_from_id(output_ids)
            output_attention_mask = torch.logical_not(torch.eq(output_ids, self.control_token_ids['output_pad_token_id']))
        elif output_embeds_enc is not None:
            # setting the output ids to unk tokens, as we don't have acces to the starting ids
            output_ids = self.control_token_ids['output_unknown_token_id'] * torch.ones(input_embeds_enc.shape[:2], dtype=torch.long).to(input_embeds_enc.device)
        else:
            # no output starting point is provided, so we start from the prepending embeddings
            output_ids = self.output_prepending_ids.repeat(input_embeds_enc.shape[0], 1).to(input_embeds_enc.device)
            output_embeds_enc = self.output_prepending_embeds_enc.repeat(input_embeds_enc.shape[0], 1, 1).to(input_embeds_enc.device)
            output_embeds_dec = self.output_prepending_embeds_dec.repeat(input_embeds_enc.shape[0], 1, 1).to(input_embeds_enc.device)
            output_attention_mask = torch.ones(output_embeds_enc.shape[:2], dtype=torch.bool).to(output_embeds_enc.device)
        
        if not teacher_force_output:
            outputs = self._sequential_forward_from_embed(self.model, self.output_discretizer, 
                input_embeds=input_embeds_enc, input_attention_mask=input_attention_mask,
                output_embeds_enc=output_embeds_enc, output_embeds_dec=output_embeds_dec, 
                output_attention_mask=output_attention_mask,
                max_output_length=max_output_length)
        else:
            model_outputs = self.model.forward(inputs_embeds=input_embeds_enc, attention_mask=input_attention_mask,
                        decoder_inputs_embeds=output_embeds_dec, decoder_attention_mask=output_attention_mask,
                        output_hidden_states=True, output_attentions=True,)
            
            outputs = self.output_discretizer(model_outputs['last_hidden_state'], supervision=True,
                target_ids=output_ids, target_attention_mask=output_attention_mask, average_probs=self.soft_average['word_embeds_with_scores_forward'])
            
            outputs['eos_flag'] = torch.any(torch.eq(outputs['id'], self.control_token_ids['output_eos_token_id']), dim=1).reshape(-1, 1)
            outputs['p_not_eos'] = 1 - outputs['score'][:, :, self.control_token_ids['output_eos_token_id']]
            outputs['output_attention_mask'] = output_attention_mask

        return {'id': outputs['id'], 'score': outputs['score'], 
                'quantized_vector_encoder': outputs['quantized_vector_encoder'], 'quantized_vector_decoder': outputs['quantized_vector_decoder'],
                'output_attention_mask': outputs['output_attention_mask'], 'eos_flag': outputs['eos_flag'], 'p_not_eos': outputs['p_not_eos'],
                'quantization_loss': outputs['quantization_loss']}

    def _sequential_forward_from_embed(self, model, discretizer, input_embeds, input_attention_mask, 
        output_embeds_enc, output_embeds_dec, output_attention_mask,
        max_output_length):

        quantization_loss = 0
        
        # In the first step we get the past_key_values and encoder_last_hidden_state
        current_output = \
            self._one_step_sequential_forward_from_embed(model, discretizer, input_embeds, input_attention_mask,
                                                    output_embeds_dec, output_attention_mask, last_step_states={})
        
        ids = current_output['id']
        scores = current_output['score']
        p_not_eoss = 1 - current_output['p_eos']
        eos_flags = current_output['eos_flag'].reshape(-1, 1)
        quantization_loss = quantization_loss * (current_output['quantization_loss'] * torch.logical_not(eos_flags))
        
        if self.config['use_last_step_states']:
            last_step_states={'encoder_outputs':(current_output['encoder_last_hidden_state'], current_output['hidden_state'], 
                                                 current_output['encoder_attentions'])}
        else:
            last_step_states = {}
        
        output_embeds_enc = torch.cat((output_embeds_enc, current_output['quantized_vector_encoder']), dim=1)
        output_embeds_dec = torch.cat((output_embeds_dec, current_output['quantized_vector_decoder']), dim=1)
        
        output_attention_mask = torch.cat((output_attention_mask, 
                                                    self.attention_mask(eos_flags, p_not_eoss[:, -1].reshape(-1, 1))), dim=1)

        while output_attention_mask.shape[1] < max_output_length and not torch.all(eos_flags):

            # use the last hidden state of the encoder as the input to the decoder
            if self.config['use_past_key_values']:
                last_step_states['past_key_values'] = current_output['past_key_values'] # used to be torch.logical_not(eos_flag) for gpt2-gpt2,

            current_output = \
            self._one_step_sequential_forward_from_embed(model, discretizer, input_embeds, input_attention_mask,
                                                    output_embeds_dec, output_attention_mask, 
                                                    last_step_states, )

            ids = torch.cat((ids, current_output['id']), dim=1)
            scores = torch.cat((scores, current_output['score']), dim=1)
            p_not_eoss = torch.cat((p_not_eoss, (1 - current_output['p_eos']) * p_not_eoss[:, -1].reshape(-1, 1)), dim=1)

            output_embeds_enc = torch.cat((output_embeds_enc, current_output['quantized_vector_encoder']), dim=1)
            output_embeds_dec = torch.cat((output_embeds_dec, current_output['quantized_vector_decoder']), dim=1)
            output_attention_mask = torch.cat((output_attention_mask, self.attention_mask(eos_flags, p_not_eoss[:, -1].reshape(-1, 1))), dim=1)

            eos_flags = torch.logical_or(eos_flags, current_output['eos_flag'].reshape(-1, 1))
            quantization_loss += (current_output['quantization_loss'] * torch.logical_not(eos_flags).float())

        # add pad tokens or embeds where attention mask is zero
        binary_attention_mask = output_attention_mask > 0
        ids = ids * binary_attention_mask[:, -ids.shape[1]:] + self.control_token_ids['output_pad_token_id'] * torch.logical_not(binary_attention_mask)[:, -ids.shape[1]:]
        scores = scores * output_attention_mask[:, -ids.shape[1]:].unsqueeze(-1) + \
            (1 - output_attention_mask[:, -ids.shape[1]:]).unsqueeze(-1) * self.onehot_score_pad
        p_not_eoss = p_not_eoss * output_attention_mask[:, -ids.shape[1]:] + (1 - output_attention_mask[:, -ids.shape[1]:]) * p_not_eoss[:, -1].reshape(-1, 1)
        output_embeds_enc = output_embeds_enc * output_attention_mask.unsqueeze(-1) + self.pad_embed_enc * torch.logical_not(output_attention_mask).unsqueeze(-1)
        output_embeds_dec = output_embeds_dec * output_attention_mask.unsqueeze(-1) + self.pad_embed_dec * torch.logical_not(output_attention_mask).unsqueeze(-1)


        
        return {
            'id': ids, 'score': scores, 'logit': current_output['logit'], 
            'quantized_vector_encoder': output_embeds_enc, 'quantized_vector_decoder': output_embeds_dec,
            'quantization_loss': quantization_loss, 'output_attention_mask': output_attention_mask, 'eos_flag': eos_flags, 'p_not_eos': p_not_eoss
        }

    def _one_step_sequential_forward_from_embed(self, model, discretizer, input_embeds, input_attention_mask, 
                                    output_embeds, output_attention_mask, last_step_states={},):
        
        eos_token_id = self.control_token_ids['output_eos_token_id']
        # If you're using cached key values or encoder outputs or what not, you should pass them here. and you should check
        # the model source code to see if the gradients backpropagate through these values from next step or not.
        output = model(inputs_embeds=input_embeds, attention_mask=input_attention_mask,
                        decoder_inputs_embeds=output_embeds, decoder_attention_mask=output_attention_mask,
                        output_hidden_states=True, output_attentions=True, **last_step_states, use_cache=self.config['use_last_step_states'])
        
        # output_embed = output['decoder_hidden_states'][-1]
        output_embed = output['last_hidden_state'][:, -1:, :] # does the same thing as above size: (batch_size, 1, hidden_size)
        # output of the encoder to be used in generation, I don't get why the key values are needed though, only query values are useful
        # maybe using this is blocking the gradient flow, I should check this
        encoder_last_hidden_state = output.encoder_last_hidden_state
        past_key_values = output['past_key_values']
        hidden_state = output.encoder_hidden_states
        encoder_attentions = output.encoder_attentions

        discretizer_output = discretizer(output_embed, supervision=False, average_probs=self.soft_average['word_embeds_with_scores_forward'])
        # discretizer_output.keys(): idx, score, logits, quantized_vector, quantization_loss
        
        current_eos_flag = (torch.eq(discretizer_output['id'][:, -1], eos_token_id))

        p_eos = discretizer_output['score'][:, :, eos_token_id]

        # idx, score, logits, quantized_vector, quantization_loss, current_eos_flag, p_eos, past_key_values, encoder_last_hidden_state, hidden_state, encoder_attentions
        return({'id': discretizer_output['id'], 'score': discretizer_output['score'], 'logit': discretizer_output['logit'], 
            'quantized_vector_encoder': discretizer_output['quantized_vector_encoder'], 'quantized_vector_decoder': discretizer_output['quantized_vector_decoder'], 
            'quantization_loss': discretizer_output['quantization_loss'], 'eos_flag': current_eos_flag, 'p_eos': p_eos, 
            'past_key_values': past_key_values, 'encoder_last_hidden_state': encoder_last_hidden_state, 
            'hidden_state': hidden_state, 'encoder_attentions': encoder_attentions})

    def attention_mask(self, eos_flags, p_not_eos):
        true_attention_mask = torch.logical_not(eos_flags)
        if self.config['soft_average']['p_eos_forward']:
            return p_not_eos
        elif self.config['soft_average']['p_eos_backward']:
            return true_attention_mask + p_not_eos - p_not_eos.detach()
        else:
            return true_attention_mask
        # what about eos_flags * p_not_eos?



def main():
    # an example for the encoder-decoder MBART model:
    # get the models and the discretizers
    unwrapped_model = UnwrappedMbart()
    vector_model = unwrapped_model['vector_model']
    en_discretizer = unwrapped_model['discretizer_enc']
    fr_discretizer = unwrapped_model['discretizer_dec']
    
    from transformers import MBart50TokenizerFast
    tokenizer = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50-many-to-many-mmt", src_lang="en_XX", tgt_lang="fr_XX")

    # prefix_ids_fr = tokenizer(text_target="", return_tensors="pt")['input_ids']
    # tokenizer.vocab['</s>']: 2, tokenizer.vocab['en_XX']: 250004, tokenizer.vocab['fr_XX']: 250008
    prefix_ids_fr = torch.tensor([2, 250008]).unsqueeze(0)

    config = {'device': 'cpu',
            'use_past_key_values': False, 'use_last_step_states': True,
            'max_lengths': {'input': 30, 'output': 30,},
            'control_token_ids': { 'input_pad_token_id': tokenizer.pad_token_id,
                                    'output_eos_token_id': tokenizer.eos_token_id, 
                                    'output_pad_token_id': tokenizer.pad_token_id,
                                    'output_unknown_token_id': tokenizer.unk_token_id,},
            'soft_average': {'p_eos_backward': True, 'p_eos_forward': False ,'word_embeds_with_scores_forward': True,},
            'output_prepending_ids': prefix_ids_fr
            }
    
    enfr_autoreg_wrapped_model = AutoRegWrapper(vector_model, en_discretizer, fr_discretizer, config)

    # an example input and output sequences
    en_batch = ['Everything that is lost that is lost.', 'we must imagine Sisyphe happy.']
    input_en = tokenizer(text=en_batch, return_tensors="pt", padding=True)
    input_ids_en = input_en['input_ids']
    # output_en_fr = enfr_autoreg_wrapped_model(input_ids=input_ids_en, input_attention_mask=None, input_embeds_enc=None,
    #                                             teacher_force_output=False)
    # # print the output of the model
    # print('--'*20)
    # print('auto-regressive forward pass - starting from the prepending embeddings (bos!)')
    # print('decoded output:', tokenizer.batch_decode(output_en_fr['id'], skip_special_tokens=False))

    # another example, starting from half of the output instead of the prepending embeddings
    sequence_fr_1 = "Tout ce qui n'est pas sauvé sera perdu."
    sequence_fr_2 = "Il faut imaginer Sisyphe heureux."
    fr_batch = [sequence_fr_1, sequence_fr_2]
    output_ids_fr = tokenizer(text_target=fr_batch, return_tensors="pt", padding=True)['input_ids'][:, 1:5]
    output_ids_fr = torch.cat((prefix_ids_fr.repeat(2, 1), output_ids_fr), axis=1)
    output_en_fr = enfr_autoreg_wrapped_model(input_ids=input_ids_en, output_ids=output_ids_fr, 
                                                teacher_force_output=False)
    # print the output of the model
    print('--'*20)
    print('auto-regressive forward pass - starting from half of the output')
    print('decoded input:', tokenizer.batch_decode(output_ids_fr, skip_special_tokens=False))
    print('decoded output:', tokenizer.batch_decode(output_en_fr['id'], skip_special_tokens=False))

    # # another example, teacher forcing the output
    # fr_batch = [sequence_fr_1, sequence_fr_2]
    # output_ids_fr = tokenizer(text_target=fr_batch, return_tensors="pt", padding=True)['input_ids'][:, 1:]
    # output_ids_fr = torch.cat((prefix_ids_fr.repeat(2, 1), output_ids_fr), axis=1)
    # output_en_fr = enfr_autoreg_wrapped_model(input_ids=input_ids_en, output_ids=output_ids_fr, 
    #                                             teacher_force_output=True)
    # # print the output of the model
    # print('--'*20)
    # print('teacher forced forward pass - teacher forcing the output')
    # print('decoded input:', tokenizer.batch_decode(output_ids_fr, skip_special_tokens=False))
    # print('decoded output:', tokenizer.batch_decode(output_en_fr['id'], skip_special_tokens=False))

    # another example, French to English translation without teacher forcing
    # prefix_ids_en = torch.tensor([2, 250004]).unsqueeze(0)
    # config['output_prepending_ids'] = prefix_ids_en
    # fren_autoreg_wrapped_model = AutoRegWrapper(vector_model, fr_discretizer, en_discretizer, config)
    # sequence_fr_1 = "Tout ce qui n'est pas sauvé sera perdu."
    # sequence_fr_2 = "Il faut imaginer Sisyphe heureux."
    # fr_batch = [sequence_fr_1, sequence_fr_2]
    # input_ids_fr = tokenizer(text_target=fr_batch, return_tensors="pt", padding=True)['input_ids']
    # output_fr_en = fren_autoreg_wrapped_model(input_ids=input_ids_fr, teacher_force_output=False)
    # # print the output of the model
    # print('--'*20)
    # print('auto-regressive forward pass - French to English translation')
    # print('decoded output:', tokenizer.batch_decode(output_fr_en['id'], skip_special_tokens=False))

if __name__ == "__main__":
    main()
