from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import MarianMTModel, MarianTokenizer
import torch
from concurrent.futures import ThreadPoolExecutor
import asyncio

app = FastAPI()

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load models and tokenizers once
models = {
    'zh_vi': MarianMTModel.from_pretrained('../../model/zh-vi').to(device),
    'zh_en': MarianMTModel.from_pretrained('../../model/zh-en').to(device),
    'en_vi': MarianMTModel.from_pretrained('../../model/en-vi').to(device),
    'en_zh': MarianMTModel.from_pretrained('../../model/en-zh').to(device)
}
tokenizers = {
    'zh_vi': MarianTokenizer.from_pretrained('../../model/zh-vi'),
    'zh_en': MarianTokenizer.from_pretrained('../../model/zh-en'),
    'en_vi': MarianTokenizer.from_pretrained('../../model/en-vi'),
    'en_zh': MarianTokenizer.from_pretrained('../../model/en-zh')
}

class TranslationRequest(BaseModel):
    src_lang: str
    tag_lang: str
    texts: list
    num_beams: int = 2
    max_length: int = 50
    batch_size: int = 16

def translate_batch(batch, src_lang, tag_lang, num_beams, max_length):
    key = f"{src_lang}_{tag_lang}"
    tokenizer = tokenizers[key]
    model = models[key]

    inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True).to(device)
    with torch.no_grad():
        translated = model.generate(
            **inputs,
            num_beams=num_beams,
            max_length=max_length,
            early_stopping=True
        )
    return [tokenizer.decode(t, skip_special_tokens=True) for t in translated]

@app.post("/translate")
async def translate(request: TranslationRequest):
    src_lang = request.src_lang
    tag_lang = request.tag_lang
    texts = request.texts
    num_beams = request.num_beams
    max_length = request.max_length
    batch_size = request.batch_size

    if f"{src_lang}_{tag_lang}" not in models or f"{src_lang}_{tag_lang}" not in tokenizers:
        raise HTTPException(status_code=400, detail="Unsupported language pair")

    batches = [texts[i:i + batch_size] for i in range(0, len(texts), batch_size)]
    results = []

    loop = asyncio.get_event_loop()
    with ThreadPoolExecutor(max_workers=4) as executor:
        tasks = [loop.run_in_executor(executor, translate_batch, batch, src_lang, tag_lang, num_beams, max_length) for batch in batches]
        results = await asyncio.gather(*tasks)

    flattened_results = [item for sublist in results for item in sublist]
    return { "status_code" : 200 ,"data": "".join(flattened_results)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=9380, workers=4)
