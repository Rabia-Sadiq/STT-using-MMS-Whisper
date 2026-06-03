from transformers import AutoProcessor

processor = AutoProcessor.from_pretrained("facebook/mms-1b-all")

for lang in ["urd-script_arabic", "pan", "eng"]:
    try:
        processor.tokenizer.set_target_lang(lang)
        print(f"  {lang} → SUPPORTED")
    except:
        print(f"  {lang} → NOT FOUND")