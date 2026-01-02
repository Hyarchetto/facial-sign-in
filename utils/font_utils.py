from PIL import ImageFont

def get_chinese_font(size=30):
    font_paths = [
        "simhei.ttf",
        "msyh.ttc",
        "simsun.ttc",
        "arialuni.ttf"
    ]
    for path in font_paths:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default()