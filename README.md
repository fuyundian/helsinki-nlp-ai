# 翻译模型安装

1. **Helsinki-NLP/opus-mt-zh-en**
    1. 安装python3编译器
        
        ```bash
        sudo apt update
        sudo apt install python3-pip
        ```
        
    2. python 安装transformers torch
        
        ```bash
        pip3 install transformers torch
        ```
        
    3. python 安装 sentencepiece
        
        ```bash
         sudo apt-get install cmake build-essential pkg-config libgoogle-perftools-dev
         pip install sentencepiece
         pip install sacremoses
        ```
        
    4. 创建python调用翻译文件（translate.py）
        
        ```python
        from transformers import MarianMTModel, MarianTokenizer
        import sys
        
        def translate(text, src_lang, tgt_lang):
            model_name = f'Helsinki-NLP/opus-mt-{src_lang}-{tgt_lang}'
            # 'Helsinki-NLP/opus-mt-en-vi'
            tokenizer = MarianTokenizer.from_pretrained(model_name)
            model = MarianMTModel.from_pretrained(model_name)
        
            translated = model.generate(**tokenizer(text, return_tensors="pt", padding=True))
            result = [tokenizer.decode(t, skip_special_tokens=True) for t in translated]
            return result[0]
        
        if __name__ == "__main__":
            src_lang = sys.argv[1]
            tgt_lang = sys.argv[2]
            text = sys.argv[3]
            translated_text = translate(text,src_lang,tgt_lang)
            print(translated_text)
        ```
        
    5. 执行命令
        
        ```bash
        python3 [translate.py](http://translate.py/) en vi  "Poetry and distant places are both an infinite yearning for a peaceful rural life and an unremitting pursuit of an ideal life. They are like a spiritual haven, allowing us to find tranquility and peace in the busy world; they are like a lighthouse in the distance, guiding us to move forward bravely and pursue a better future. In the poetic words, we feel the expression of emotions, and among the green mountains and clear waters in the distance, we explore the true meaning of life and ultimately achieve a fulfilled and rich self."
        ```
        
    6. 进阶安装服务
        
        ```
        	pip install gunicorn
        	pip install Flask
        	pip install fastapi uvicorn
        ```
        
    7. 下载离线模型代码
        
        ```
        from transformers import MarianMTModel, MarianTokenizer
        
        def download_model(model_name,model_project):
            tokenizer = MarianTokenizer.from_pretrained(model_name)
            model = MarianMTModel.from_pretrained(model_name)
            tokenizer.save_pretrained(f'./model/{model_project}')
            model.save_pretrained(f'./model/{model_project}')
        
        if __name__ == "__main__":
            model_name = "Helsinki-NLP/opus-mt-en-vi"
            download_model(model_name,'en-vi')
        
        ```
        
    8. 离线模型下载
        
        ```
        python3 /mnt/d/project/by_token/download_model.py
        ```
        
    9. 编写服务端代码
        
        ```python
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
            'zh_vi': MarianMTModel.from_pretrained('Helsinki-NLP/opus-mt-zh-vi').to(device),
            'zh_en': MarianMTModel.from_pretrained('Helsinki-NLP/opus-mt-zh-en').to(device),
            'en_vi': MarianMTModel.from_pretrained('Helsinki-NLP/opus-mt-en-vi').to(device),
            'en_zh': MarianMTModel.from_pretrained('Helsinki-NLP/opus-mt-en-zh').to(device)
        }
        tokenizers = {
            'zh_vi': MarianTokenizer.from_pretrained('Helsinki-NLP/opus-mt-zh-vi'),
            'zh_en': MarianTokenizer.from_pretrained('Helsinki-NLP/opus-mt-zh-en'),
            'en_vi': MarianTokenizer.from_pretrained('Helsinki-NLP/opus-mt-en-vi'),
            'en_zh': MarianTokenizer.from_pretrained('Helsinki-NLP/opus-mt-en-zh')
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
        
        ```
        
    10. 执行服务端代码
        
        ```bash
         uvicorn src/main/translate_server_v2:app --host 0.0.0.0 --port 9380 --workers 8
        ```
        
    11. 客户端调用
        
        ```bash
        curl -X POST "http://localhost:9380/translate" -H "Content-Type: application/json" -d '{
          "src_lang": "zh",
          "tgt_lang": "en",
          "texts": "你好,
        }'
        ```