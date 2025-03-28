class BertTokenizerStandalone:
    """Simple standalone implementation of BertTokenizer basics"""
    def __init__(self, vocab_file=None):
        # Use a small basic vocabulary for demonstration
        self.vocab = {"[PAD]": 0, "[CLS]": 1, "[SEP]": 2, "[UNK]": 3}
        self.vocab.update({f"token{i}": i+4 for i in range(30000)})  # Dummy vocab
        self.ids_to_tokens = {v: k for k, v in self.vocab.items()}
        
    def tokenize(self, text):
        # Very simplified tokenization - just split by space
        return text.lower().split()
    
    def convert_tokens_to_ids(self, tokens):
        return [self.vocab.get(token, self.vocab["[UNK]"]) for token in tokens]
    
    def encode(self, text, text_pair=None, max_length=None, truncation=False, padding=False):
        tokens = ["[CLS]"] + self.tokenize(text)
        if text_pair:
            tokens += ["[SEP]"] + self.tokenize(text_pair)
        else:
            tokens += ["[SEP]"]
            
        if truncation and max_length and len(tokens) > max_length:
            tokens = tokens[:max_length-1] + ["[SEP]"]
            
        token_ids = self.convert_tokens_to_ids(tokens)
        
        if padding and max_length:
            padding_length = max_length - len(token_ids)
            token_ids = token_ids + [0] * padding_length
            
        return token_ids
    
    def __call__(self, text, text_pair=None, max_length=None, truncation=False, padding=False, return_tensors=None):
        if isinstance(text, str):
            token_ids = self.encode(text, text_pair, max_length, truncation, padding)
            attention_mask = [1] * len(token_ids)
            if padding and max_length:
                attention_mask = [1] * (len(token_ids) - attention_mask.count(0)) + [0] * attention_mask.count(0)
                
            token_type_ids = [0] * len(token_ids)
            if text_pair:
                sep_pos = token_ids.index(self.vocab["[SEP]"])
                token_type_ids[sep_pos+1:] = [1] * (len(token_ids) - sep_pos - 1)
                
            result = {
                "input_ids": token_ids,
                "attention_mask": attention_mask,
                "token_type_ids": token_type_ids
            }
            
            if return_tensors == "pt":
                result = {k: torch.tensor([v]) for k, v in result.items()}
                
            return result
        else:
            # For batch processing
            results = []
            for t, tp in zip(text, text_pair if text_pair else [None] * len(text)):
                results.append(self(t, tp, max_length, truncation, padding))
                
            # Combine results
            batch_result = {
                "input_ids": [],
                "attention_mask": [],
                "token_type_ids": []
            }
            
            for r in results:
                batch_result["input_ids"].append(r["input_ids"])
                batch_result["attention_mask"].append(r["attention_mask"])
                batch_result["token_type_ids"].append(r["token_type_ids"])
                
            if return_tensors == "pt":
                batch_result = {k: torch.tensor(v) for k, v in batch_result.items()}
                
            return batch_result 