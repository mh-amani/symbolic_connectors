from typing import Any, Dict, Tuple
import torch
from torch.nn import ModuleDict, Module
from transformers import MBart50TokenizerFast  
from blocks.unwrapped_models.enc_dec_unwrapper import UnwrappedMbart


class BlocksConnector(Module):
    """
    a wrapper connecting two sequence models with discrete bottleneck layers
    """
    def __init__(
        self,
        vector_model_x_to_y,
        vector_model_y_to_z,
        discretizer_x,
        discretizer_y,
        discretizer_z,
        config= None,
    ) -> None:

        super().__init__()
        self.config = config
        self.model_x_to_y = vector_model_x_to_y
        self.model_y_to_z = vector_model_y_to_z
        self.discretizer_x = discretizer_x
        self.discretizer_y = discretizer_y
        self.discretizer_z = discretizer_z
        self.max_lengths = self.config['max_lengths']
        self.control_token_ids = self.config['control_token_ids']
        self.soft_average = self.config['soft_average']

    def forward(self, x_ids: torch.Tensor=None, x_attention_mask: torch.Tensor=None, x_embeds_enc: torch.Tensor=None,
                      y_prepending_ids: torch.Tensor=None, y_prepending_embeds_enc: torch.Tensor=None, y_prepending_embeds_dec: torch.Tensor=None,
                      z_ids: torch.Tensor=None, z_attention_mask: torch.Tensor=None, z_embeds_dec: torch.Tensor=None,
                      teacher_force_z: bool=True) -> Tuple[torch.Tensor, torch.Tensor]:
        """Perform a forward pass through the models x -> y -> z

        :param x: A tensor of images.
        :return: A tensor of logits.
        """
        # assert that only one of the embeds or ids is provided, and attention mask should be provided if embeds are provided
        assert (x_ids is not None) != (x_embeds_enc is not None), "Either x_ids or x_embeds should be provided"
        # assert (z_ids is not None) != (z_embeds_dec is not None), "Either z_ids or z_embeds should be provided"
        assert (x_embeds_enc is not None and x_attention_mask is not None) or (x_embeds_enc is None and x_attention_mask is None), "x_embeds and x_attention_mask should be provided together or not at all"
        assert (z_embeds_dec is not None and z_attention_mask is not None) or (z_embeds_dec is None and z_attention_mask is None), "z_embeds and z_attention_mask should be provided together or not at all"
        
        teacher_force_z = teacher_force_z and (z_embeds_dec or z_ids is not None)
        
        # warn if y_prepending is not provided, and start with BOS token
        if y_prepending_ids is None and (y_prepending_embeds_enc is None or y_prepending_embeds_dec is None):
            print("Warning: y_prepending_ids nor the embeddings are not provided, starting with BOS token")
            y_prepending_ids = self.control_token_ids['bos_token_id_y'] * torch.ones(x_ids.shape[0], 1, dtype=torch.long).to(x_ids.device)
        elif y_prepending_ids is None:
            y_prepending_ids = self.control_token_ids['pad_token_id_y'] * torch.ones(y_prepending_embeds_dec.shape[:2], dtype=torch.long).to(y_prepending_embeds_dec.device)       
        
        if x_ids is not None:
            x_embeds_enc = self.discretizer_x.encoder_embedding_from_id(x_ids)
            x_attention_mask = torch.logical_not(torch.eq(x_ids, self.control_token_ids['pad_token_id_x']))
        if y_prepending_ids is not None:
            y_prepending_embeds_enc = self.discretizer_y.encoder_embedding_from_id(y_prepending_ids)
            y_prepending_embeds_dec = self.discretizer_y.decoder_embedding_from_id(y_prepending_ids)
        if z_ids is not None:
            z_embeds_dec = self.discretizer_z.decoder_embedding_from_id(z_ids)
            z_attention_mask = torch.logical_not(torch.eq(z_ids, self.control_token_ids['pad_token_id_z']))
        # y_prepending mask is ones! 
        y_prepending_attention_mask = torch.ones(*y_prepending_embeds_dec.shape[:2], device=y_prepending_embeds_enc.device, dtype=torch.bool)
        
        xy_outputs = self._sequential_forward_from_embed(model=self.model_x_to_y, discretizer=self.discretizer_y, 
            input_sequence_embeds=x_embeds_enc, input_sequence_attention_mask=x_attention_mask,
            output_sequence_embeds_enc=y_prepending_embeds_enc, output_sequence_embeds_dec=y_prepending_embeds_dec, 
            output_sequence_attention_mask=y_prepending_attention_mask, output_sequence_ids=y_prepending_ids,
            max_output_length=self.max_lengths['y'], eos_token_id=self.control_token_ids['eos_token_id_y'], pad_token_id=self.control_token_ids['pad_token_id_y'],)

        if teacher_force_z:
            yz_outputs = self.model_y_to_z(inputs_embeds=xy_outputs['quantized_vector_encoder'], attention_mask=xy_outputs['output_attention_mask'],
                                        decoder_inputs_embeds=z_embeds_dec, decoder_attention_mask=z_attention_mask, return_dict=True)
        
            yz_discretized_output = self.discretizer_z(yz_outputs['last_hidden_state'], supervision=True, true_ids=z_ids[:, 1:])
            z_quantization_loss = yz_discretized_output['quantization_loss'] * z_attention_mask / z_attention_mask[:, 1:].sum()
        else:
            yz_outputs = self.sequential_forward(model=self.model_y_to_z, discretizer=self.discretizer_z,
                input_sequence_embeds=xy_outputs['quantized_vector_encoder'], input_sequence_attention_mask=xy_outputs['output_attention_mask'],
                output_sequence_embeds_enc=xy_outputs['quantized_vector_decoder'], output_sequence_embeds_dec=xy_outputs['quantized_vector_decoder'],
                output_sequence_attention_mask=xy_outputs['output_attention_mask'], output_sequence_ids=z_ids,
                max_output_length=self.max_lengths['z'], eos_token_id=self.control_token_ids['eos_token_id_z'], pad_token_id=self.control_token_ids['pad_token_id_z'],)
            z_quantization_loss = yz_outputs['quantization_loss']

        quantization_loss = z_quantization_loss + yz_discretized_output['quantization_loss']
        
        return { 'id_y': xy_outputs['id'], 'id_z': yz_discretized_output['id'], 
                'score_y': xy_outputs['score'], 'score_z': yz_discretized_output['score'],
                'quantization_loss': quantization_loss}

    
    def autoregressive_forward(self, model, discretizer_input, discretizer_output, 
                    input_ids: torch.Tensor=None, input_attention_mask: torch.Tensor=None, input_embeds_enc: torch.Tensor=None,
                    output_prepending_ids: torch.Tensor=None, output_prepending_embeds_enc: torch.Tensor=None, 
                    output_prepending_embeds_dec: torch.Tensor=None,
                    max_out_length=None, 
                    output_bos_token_id=None, output_eos_token_id=None, output_pad_token_id=None, input_pad_token_id=None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Perform a forward pass through the models x -> y_1 y_2 ... y_n

        """
        # if token ids are not provided, use the bos token for the first step
        if max_out_length is None:
            max_out_length = self.max_lengths['y']
        if output_bos_token_id is None and (output_prepending_ids is None and output_prepending_embeds_enc is None):
            output_bos_token_id = self.control_token_ids['bos_token_id_y']
        if output_eos_token_id is None:
            output_eos_token_id = self.control_token_ids['eos_token_id_y']
        if output_pad_token_id is None:
            output_pad_token_id = self.control_token_ids['pad_token_id_y']
        if input_pad_token_id is None:
            input_pad_token_id = self.control_token_ids['pad_token_id_x']
        # assert that only one of the embeds or ids is provided, and attention mask should be provided if embeds are provided
        assert (input_ids is not None) != (input_embeds_enc is not None), "Either input_ids or input_embeds should be provided"
        assert (input_embeds_enc is not None and input_attention_mask is not None) or (input_embeds_enc is None and input_attention_mask is None), "input_embeds and input_attention_mask should be provided together or not at all"
        
        # warn if y_prepending is not provided, and start with BOS token
        if output_prepending_ids is None and (y_output_prepending_embeds_enc is None or output_prepending_embeds_dec is None):
            print("Warning: output_prepending_ids nor the embeddings are not provided, starting with BOS token")
            output_prepending_ids = self.control_token_ids['bos_token_id_y'] * torch.ones(input_ids.shape[0], 1, dtype=torch.long).to(input_ids.device)
        elif output_prepending_ids is None:
            output_prepending_ids = self.control_token_ids['pad_token_id_y'] * torch.ones(output_prepending_embeds_dec.shape[:2], dtype=torch.long).to(output_prepending_embeds_dec.device)       
        
        if input_ids is not None:
            input_embeds_enc = self.discretizer_x.encoder_embedding_from_id(input_ids)
            input_attention_mask = torch.logical_not(torch.eq(input_ids, input_pad_token_id))
        if output_prepending_ids is not None:
            output_prepending_embeds_enc = self.discretizer_y.encoder_embedding_from_id(output_prepending_ids)
            output_prepending_embeds_dec = self.discretizer_y.decoder_embedding_from_id(output_prepending_ids)
        
        # y_prepending mask is ones! 
        output_prepending_attention_mask = torch.ones(*output_prepending_embeds_dec.shape[:2], device=output_prepending_embeds_enc.device, dtype=torch.bool)
        
        outputs = self._sequential_forward_from_embed(model, discretizer_output, 
            input_sequence_embeds=input_embeds_enc, input_sequence_attention_mask=input_attention_mask,
            output_sequence_embeds_enc=output_prepending_embeds_enc, output_sequence_embeds_dec=output_prepending_embeds_dec, 
            output_sequence_attention_mask=output_prepending_attention_mask, output_sequence_ids=output_prepending_ids,
            max_output_length=max_out_length, eos_token_id=output_eos_token_id, pad_token_id=output_pad_token_id,)
        print(outputs.keys())
        return {'id': outputs['id'], 'score': outputs['score'], 
                'quantized_vector_encoder': outputs['quantized_vector_encoder'], 'quantized_vector_decoder': outputs['quantized_vector_decoder'],
                'attention_mask': outputs['output_attention_mask'], 'eos_flag': outputs['eos_flag'], 'p_not_eos': outputs['p_not_eos'],
                'quantization_loss': outputs['quantization_loss']}

    def _sequential_forward_from_embed(self, model, discretizer, input_sequence_embeds, input_sequence_attention_mask, 
        output_sequence_embeds_enc, output_sequence_embeds_dec, output_sequence_attention_mask, output_sequence_ids,
        max_output_length, eos_token_id, pad_token_id,):

        if max_output_length is None:
            max_output_length = self.max_output_length - 3
        
        past_key_values = None
        quantization_loss = 0
        # eos_flag is a vector showing if the eos token has been generated for the sequences in the batch
        # output_attention_mask is a matrix showing which tokens are valid to attend to in the output sequence, basically
        # all tokens except the ones that are generated after the eos token (substituted with paddings)

        # In the first step we get the past_key_values and encoder_last_hidden_state
        current_output = \
            self.one_step_sequential_forward(model, discretizer, input_sequence_embeds, input_sequence_attention_mask,
                                                    output_sequence_embeds_dec, output_sequence_attention_mask)
        # current_output.keys(): id', 'score', 'logit', 'quantized_vector_encoder', 'quantized_vector_decoder', 
        # 'quantization_loss', 'eos_flag', 'p_eos', 'past_key_values', 'encoder_last_hidden_state', 'hidden_state', 'encoder_attentions'
        ids = current_output['id']
        scores = current_output['score']
        p_not_eoss = 1 - current_output['p_eos']
        eos_flags = current_output['eos_flag'].reshape(-1, 1)
        quantization_loss = quantization_loss * (current_output['quantization_loss'] * torch.logical_not(eos_flags))
        
        if self.config['use_last_step_states']:
            last_step_states={'encoder_outputs':(current_output['encoder_last_hidden_state'], current_output['hidden_state'], current_output['encoder_attentions'])}
        else:
            last_step_states = {}
        counter = 1
        
        output_sequence_embeds_enc = torch.cat((output_sequence_embeds_enc, current_output['quantized_vector_encoder']), dim=1)
        output_sequence_embeds_dec = torch.cat((output_sequence_embeds_dec, current_output['quantized_vector_decoder']), dim=1)
        
        output_sequence_attention_mask = torch.cat((output_sequence_attention_mask, self.attention_mask(eos_flags, p_not_eoss[:, -1].reshape(-1, 1))), dim=1)

        while output_sequence_attention_mask.shape[1] < max_output_length and not torch.all(eos_flags):

            # use the last hidden state of the encoder as the input to the decoder
            if self.config['use_past_key_values']:
                last_step_states['past_key_values'] = current_output['past_key_values'] # used to be torch.logical_not(eos_flag) for gpt2-gpt2,

            current_output = \
            self.one_step_sequential_forward(model, discretizer, input_sequence_embeds, input_sequence_attention_mask,
                                                    output_sequence_embeds_dec, output_sequence_attention_mask, last_step_states)

            ids = torch.cat((ids, current_output['id']), dim=1)
            scores = torch.cat((scores, current_output['score']), dim=1)
            p_not_eoss = torch.cat((p_not_eoss, (1 - current_output['p_eos']) * p_not_eoss[:, -1].reshape(-1, 1)), dim=1)
            eos_flags = torch.logical_or(eos_flags, current_output['eos_flag'].reshape(-1, 1))
            quantization_loss += (current_output['quantization_loss'] * torch.logical_not(eos_flags).float())

            output_sequence_embeds_enc = torch.cat((output_sequence_embeds_enc, current_output['quantized_vector_encoder']), dim=1)
            output_sequence_embeds_dec = torch.cat((output_sequence_embeds_dec, current_output['quantized_vector_decoder']), dim=1)
            output_sequence_attention_mask = torch.cat((output_sequence_attention_mask, self.attention_mask(eos_flags, p_not_eoss[:, -1].reshape(-1, 1))), dim=1)
        
        return {
            'id': ids, 'score': scores, 'logit': current_output['logit'], 
            'quantized_vector_encoder': output_sequence_embeds_enc, 'quantized_vector_decoder': output_sequence_embeds_dec,
            'quantization_loss': quantization_loss, 'output_attention_mask': output_sequence_attention_mask, 'eos_flag': eos_flags, 'p_not_eos': p_not_eoss
        }

    def one_step_sequential_forward(self, model, discretizer, input_embeds, input_attention_mask, 
                                    output_embeds, output_attention_mask, last_step_states={}):
        
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
        
        current_eos_flag = (torch.eq(discretizer_output['id'][:, -1], self.control_token_ids['eos_token_id_y']))

        p_eos = discretizer_output['score'][:, :, self.control_token_ids['eos_token_id_y']]

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
    model_enfr = unwrapped_model['model']
    vector_model = unwrapped_model['vector_model']
    en_discretizer = unwrapped_model['discretizer_enc']
    fr_discretizer = unwrapped_model['discretizer_z']
    
    from transformers import MBart50TokenizerFast
    tokenizer = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50-many-to-many-mmt", src_lang="en_XX", tgt_lang="fr_XX")

    config ={'use_past_key_values': False, 'use_last_step_states': True,
        'max_lengths': {'x': 15, 'y': 15, 'z': 15},
        'control_token_ids': { 'pad_token_id_x': tokenizer.pad_token_id,
                                'eos_token_id_y': tokenizer.eos_token_id, 
                                'pad_token_id_y': tokenizer.pad_token_id, 
                                'pad_token_id_z': tokenizer.pad_token_id},
        'soft_average': {'p_eos_backward': True, 'p_eos_forward': False ,'word_embeds_with_scores_forward': True,}
        }
    en_fr_en_connected_models = BlocksConnector(vector_model, vector_model, en_discretizer, fr_discretizer, en_discretizer, config=config)

    # an example input and output sequences
    sequence_en_1 = "Everything not saved will be lost."
    sequence_en_2 = "one must imagine Sisyphus happy."
    sequence_fr_1= "Tout ce qui n'est pas sauvé sera perdu." # "il m'a entarté"
    sequence_fr_2 = "il faut imaginer Sisyphe heureux."
    en_batch = [sequence_en_1, sequence_en_2]
    fr_batch = [sequence_fr_1, sequence_fr_2]
    input_en = tokenizer(text=en_batch, return_tensors="pt", padding=True)
    output_fr = tokenizer(text_target=fr_batch, return_tensors="pt", padding=True) # the text_target is not doing anything!
    input_ids_en, input_attention_mask_en = input_en['input_ids'], input_en['attention_mask']
    output_ids_fr, output_attention_mask_fr = output_fr['input_ids'], output_fr['attention_mask']
    # add <EOS> to the beginning of labels ONLY FOR MBART CAUSE IT'S SO WEIRD
    output_ids_fr = torch.cat((torch.ones((output_ids_fr.shape[0], 1), dtype=torch.long).to(output_ids_fr.device)*tokenizer.eos_token_id, output_ids_fr), dim=1)
    output_attention_mask_fr = torch.cat((torch.ones((output_attention_mask_fr.shape[0], 1), dtype=torch.long).to(output_attention_mask_fr.device), output_attention_mask_fr), dim=1)
    prefix_ids_fr = output_ids_fr[:, :2]
    prefix_attention_mask_fr = output_attention_mask_fr[:, :2]

    output_en_fr = en_fr_en_connected_models.autoregressive_forward(
        vector_model, en_discretizer, fr_discretizer, 
        input_ids=input_ids_en, input_attention_mask=None, input_embeds_enc=None,
        output_prepending_ids=prefix_ids_fr, output_prepending_embeds_enc=None, 
        output_prepending_embeds_dec=None,
        max_out_length=15, 
        output_bos_token_id=None, output_eos_token_id=config['control_token_ids']['eos_token_id_y'], output_pad_token_id=config['control_token_ids']['pad_token_id_y'], input_pad_token_id=config['control_token_ids']['pad_token_id_x'])
    # print the output of the model
    print('decoded output:', tokenizer.batch_decode(output_en_fr['id'], skip_special_tokens=False))

    output_en_fr_en = en_fr_en_connected_models(x_ids=input_ids_en, y_prepending_ids=prefix_ids_fr, z_ids=input_ids_en, teacher_force_z=True)
    # print the output of the model
    print('decoded output:', tokenizer.batch_decode(output_en_fr_en['id_y'], skip_special_tokens=False))
    print('decoded output:', tokenizer.batch_decode(output_en_fr_en['id_z'], skip_special_tokens=False))



if __name__ == "__main__":
    main()
