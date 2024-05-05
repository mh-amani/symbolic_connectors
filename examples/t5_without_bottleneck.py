from datasets import load_dataset

books = load_dataset("opus_books", "en-fr")
books = books["train"].train_test_split(test_size=0.2)
print('example of dataset elements', books["train"][0])

from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("t5-small")

from transformers import T5Tokenizer, T5ForConditionalGeneration

    tokenizer = T5Tokenizer.from_pretrained("google-t5/t5-small")
    model = T5ForConditionalGeneration.from_pretrained("google-t5/t5-small")

    # the following 2 hyperparameters are task-specific
    max_en_length = 256
    max_fr_length = 256

    # Suppose we have the following 2 training examples:
    input_sequence_en_1 = "Welcome to NYC"
    input_sequence_en_2 = "HuggingFace is a company"
    input_sequence_fr_1= "Bienvenue à NYC"
    output_sequence_fr_2 = "HuggingFace est une entreprise"

    # encode the inputs
    task_prefix_enfr = "translate English to French: "
    task_prefix_fren = "translate French to English: "
    input_sequences_en = [input_sequence_en_1, input_sequence_en_2]
    input_sequences_fr = [input_sequence_fr_1, output_sequence_fr_2]

    encoding_en = tokenizer(
        [task_prefix_enfr + sequence for sequence in input_sequences_en],
        padding="longest",
        max_length=max_en_length,
        truncation=True,
        return_tensors="pt",
    )
    encoding_fr = tokenizer(
        [task_prefix_fren + sequence for sequence in input_sequences_fr],
        padding="longest",
        max_length=max_fr_length,
        truncation=True,
        return_tensors="pt",
    )
    decoding_fr = tokenizer(
        input_sequences_fr,
        padding="longest",
        max_length=max_fr_length,
        truncation=True,
        return_tensors="pt",
    )
    decoding_en = tokenizer(
        input_sequences_en,
        padding="longest",
        max_length=max_en_length,
        truncation=True,
        return_tensors="pt",
    )

    input_ids_en, attention_mask_en = encoding_en.input_ids, encoding_en.attention_mask
    input_ids_fr, attention_mask_fr = encoding_fr.input_ids, encoding_fr.attention_mask
    labels_en = decoding_en.input_ids
    labels_fr = decoding_fr.input_ids


    # replace padding token id's of the labels by -100 so it's ignored by the loss
    # labels[labels == tokenizer.pad_token_id] = -100

    # autoreg generation
    # output_enfr = model.generate(input_ids=input_ids_en, attention_mask=attention_mask_en, max_length=max_en_length)
    # tokenizer.decode(output_enfr[1])


    # forward pass
    output_enfr = model(input_ids=input_ids_en, attention_mask=attention_mask_en, labels=labels_fr)
    output_fren = model(input_ids=input_ids_fr, attention_mask=attention_mask_fr, labels=labels_en)

    loss_enfr = output_enfr.loss
    loss_fren = output_fren.loss
    print('loss_enfr:', loss_enfr.item())
    print('loss_fren:', loss_fren.item())
    print('input sentence 0: ', input_sequence_en_1, ' \n',
    'output sentence 0 :', tokenizer.decode(output_enfr.logits.argmax(-1)[0], skip_special_tokens=True), ' \n',
    'input sentence 1: ', input_sequence_en_2, ' \n',
    'output sentence 1 :', tokenizer.decode(output_enfr.logits.argmax(-1)[1], skip_special_tokens=True))
    print('input sentence 0: ', input_sequence_fr_1, ' \n',
    'output sentence 0 :', tokenizer.decode(output_fren.logits.argmax(-1)[0], skip_special_tokens=True), ' \n',
    'input sentence 1: ', output_sequence_fr_2, ' \n',
    'output sentence 1 :', tokenizer.decode(output_fren.logits.argmax(-1)[1], skip_special_tokens=True))