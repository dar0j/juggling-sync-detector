from PIL import Image
import glob
import os

output_dir = "extracted_frames"
os.makedirs(output_dir, exist_ok=True)

for filepath in glob.glob("gifs/*.gif"):
    gif = Image.open(filepath)
    base_name = os.path.splitext(os.path.basename(filepath))[0]
    for frame in range(gif.n_frames):
        gif.seek(frame)
        out_name = f"{base_name}_frame_{frame}.png"
        gif.save(os.path.join(output_dir, out_name))