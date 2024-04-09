from transformers import BertTokenizer, BertModel
import torch
from sklearn.metrics.pairwise import cosine_similarity

# Load pre-trained model tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Load pre-trained model
model = BertModel.from_pretrained('bert-base-uncased')

# Define the input text
texts = [
    "I don't find that very helpful.",
    "Why thank you!!",
]


def concatenate_last_n_hidden_states(outputs, n=4):
    last_n_hidden_states = [outputs.hidden_states[-i][:, 0, :] for i in range(1, n + 1)]
    concatenated_hidden_states = torch.cat(last_n_hidden_states, dim=-1)

    ## TEST
    # # Extract the last hidden state from the concatenated tensor
    # last_hidden_state_concatenated = concatenated_hidden_states[:, 768:768*2]
    # # Extract the last hidden state directly from outputs
    # last_hidden_state_output = outputs.hidden_states[-2][:, 0, :]
    # # Print shapes for debugging
    # print(last_hidden_state_concatenated.shape)
    # print(last_hidden_state_output.shape)

    # assert torch.all(last_hidden_state_concatenated == last_hidden_state_output)

    return concatenated_hidden_states

def four_layer_embeddings(input_texts):
    inputs = tokenizer(input_texts, padding=True, add_special_tokens=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    return concatenate_last_n_hidden_states(outputs, n=4)

if __name__ == "__main__":
    print(four_layer_embeddings("hello world!"))
