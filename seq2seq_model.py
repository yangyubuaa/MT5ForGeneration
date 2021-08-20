import torch

from mt5_model.transformers_pkg.models.t5 import T5ForConditionalGeneration
from mt5_model.transformers_pkg.models.t5 import T5Tokenizer


class MT5ForJudgementGeneration(torch.nn.Module):
    def __init__(self, config):
        super(MT5ForJudgementGeneration, self).__init__()
        self.config = config
        self.mt5_model = T5ForConditionalGeneration.from_pretrained("mt5_model/mt5-base")

    def forward(self, input_ids, labels):
        return self.mt5_model(input_ids=input_ids, labels=labels)

    def generate(self, input_ids):
        output = self.mt5_model.generate(input_ids=input_ids)
        output_token_ids = output.numpy().tolist()
        return output_token_ids


if __name__ == '__main__':
    model = MT5ForJudgementGeneration()
    tokenizer = T5Tokenizer.from_pretrained("mt5_model/mt5-base")
    input_ids = tokenizer('将中文翻译成英文：我爱你', return_tensors='pt').input_ids
    labels = tokenizer('Das Haus ist wunderbar.', return_tensors='pt').input_ids
    print(input_ids, labels)
    print(input_ids.numpy().tolist()[0])
    print(tokenizer.decode(input_ids.numpy().tolist()[0]))
    # the forward function automatically creates the correct decoder_input_ids
