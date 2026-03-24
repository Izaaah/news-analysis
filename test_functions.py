"""
Script untuk test fungsi summarize_text dan extract_keywords
"""
from model import summarize_text, extract_keywords

# Test text
test_text = """
Samsung meluncurkan smartphone baru dengan chipset AI dan kamera canggih. 
Perangkat ini dilengkapi dengan teknologi terbaru yang memungkinkan pengguna 
untuk mengambil foto berkualitas tinggi. Smartphone ini juga memiliki baterai 
yang tahan lama dan layar yang jernih. Harga yang ditawarkan cukup kompetitif 
untuk segmen pasar menengah ke atas.
"""

print("=" * 50)
print("TESTING SUMMARIZE_TEXT")
print("=" * 50)
try:
    summary = summarize_text(test_text)
    print(f"Summary: {summary}")
    print(f"Summary length: {len(summary)}")
    print("✓ summarize_text berhasil")
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 50)
print("TESTING EXTRACT_KEYWORDS")
print("=" * 50)
try:
    keywords = extract_keywords(test_text)
    print(f"Keywords: {keywords}")
    print(f"Keywords count: {len(keywords)}")
    print("✓ extract_keywords berhasil")
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 50)
print("TEST SELESAI")
print("=" * 50)



