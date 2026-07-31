from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

# try default font (no setup needed)
font = ImageFont.load_default()

for name in common_names:
    img1 = Image.open(images1[name]).convert("RGB")
    img2 = Image.open(images2[name]).convert("RGB")

    h = max(img1.height, img2.height)
    text_h = 25  # height for title

    total_w = img1.width + img2.width
    new_img = Image.new("RGB", (total_w, h + text_h), (0, 0, 0))

    draw = ImageDraw.Draw(new_img)

    # draw filename at top
    draw.text((10, 5), name, fill=(255, 255, 255), font=font)

    # paste images below text
    new_img.paste(img1, (0, text_h))
    new_img.paste(img2, (img1.width, text_h))

    out_path = output_folder / name
    new_img.save(out_path)

print("Done.")