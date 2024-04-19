"""
This program demonstrates how to use a prefix tree to guide inference.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
import pickle
import pandas as pd
import torch

from trie import Trie

if __name__ == '__main__':
    # data process
    datas = ['China', 'USA', 'Canada', 'Australia', 'UK', 'Japan', 'China']
    datas = list(set([item.strip() for item in datas]))

    # tokenize
    from transformers import AutoTokenizer, OPTForCausalLM
    tokenizer = AutoTokenizer.from_pretrained('facebook/opt-2.7b', use_fast=False) # facebook/opt-2.7b

    # Ensure that each item is surrounded by special tokens,
    # with the first token used as the unified root node of the tree,
    # and the final token using eos to guide the model to terminate generation
    datas = [f'{item}{tokenizer.eos_token}' for item in datas]
    datas_ids = [tokenizer(t)['input_ids'] for t in datas]
    # China: [2, 8481, 2]

    # build tree
    tree = Trie(datas_ids)
    prefix_tree_dict = tree.trie_dict
    with open('./prefix_tree.pkl', 'wb') as f:
        pickle.dump(prefix_tree_dict, f)

    # load tree
    trie_dict = pd.read_pickle('./prefix_tree.pkl')
    trie = Trie.load_from_dict(trie_dict, bos_token_id=tokenizer.eos_token_id)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Ensure that each input_text is end with eos
    input_text = f"China, officially the People's Republic of China (PRC), is a country in East Asia. Its population exceeding 1.4 billion makes it the world's second-most populous country. Which country is described in the text?{tokenizer.eos_token}"
    input_ids = tokenizer.encode(input_text, return_tensors='pt').to(device) # [[2, ..., 2], ]

    model = OPTForCausalLM.from_pretrained('facebook/opt-2.7b') #  torch_dtype=torch.float16
    model.to(device)

    model.eval()
    with torch.no_grad():
        fn = lambda batch_id, sent: trie.get(sent.tolist())
        output = model.generate(input_ids, prefix_allowed_tokens_fn=fn, max_new_tokens=20)
        decoded_output = tokenizer.decode(output[0], skip_special_tokens=True)
        print(decoded_output)

