import asyncio
from googletrans import Translator



async def implement_translate_text(name_list):
    async with Translator() as translator:
        tasks = [ translator.translate(text, dest='en') for text in name_list]
        results = await asyncio.gather(*tasks)
    return [res.text for res in results]

def translate_text(name_list):
    results = asyncio.run(implement_translate_text(name_list))
    return results

def main():
    name_list = ["你好", "世界", "音乐"]
    results = translate_text(name_list)
    print(results)
        

if __name__ == "__main__":
    main()
