import os
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

from transformers import AutoProcessor, Wav2Vec2ForCTC

print("Checking MMS model for Punjabi support...")
print("(Downloads tokenizer only ~few MB)")

model_id = "facebook/mms-1b-all"

processor = AutoProcessor.from_pretrained(model_id)

# Check supported languages
print(f"\nTotal languages supported: {len(processor.tokenizer.vocab)}")

# Check Punjabi specifically
languages_to_check = ["pnb", "pan", "ur", "eng"]
print("\nChecking target languages:")
for lang in languages_to_check:
    try:
        processor.tokenizer.set_target_lang(lang)
        print(f"  {lang} → SUPPORTED")
    except Exception as e:
        print(f"  {lang} → NOT FOUND ({e})")