from transformers import MarianMTModel, MarianTokenizer

def download_model(model_name,model_project):
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)
    tokenizer.save_pretrained(f'./model/{model_project}')
    model.save_pretrained(f'./model/{model_project}')

if __name__ == "__main__":
    model_name = "Helsinki-NLP/opus-mt-en-zh"
    download_model(model_name,'en-zh')
