from pathlib import Path
import random, shutil
from PIL import Image, ImageDraw, ImageFilter

SEED = 20261020
random.seed(SEED)
ROOT = Path('screen_status_dataset')
IMG_W, IMG_H = 640, 480
CLASSES = ['normal', 'occluded', 'blurred', 'black_screen']

def make_base():
    im = Image.new('RGB', (IMG_W, IMG_H), tuple(random.randint(25, 70) for _ in range(3)))
    d = ImageDraw.Draw(im)
    # desk / wall background and a monitor-like bezel
    d.rectangle((20, 30, 620, 450), fill=(15, 18, 22), outline=(90, 95, 105), width=5)
    d.rectangle((45, 55, 595, 405), fill=(18, 25, 34))
    # generated UI content
    accent = random.choice([(40, 150, 240), (80, 210, 150), (240, 160, 50), (180, 100, 230)])
    d.rectangle((45, 55, 595, 90), fill=accent)
    for y in range(115, 370, 48):
        d.rounded_rectangle((70, y, random.randint(260, 520), y+25), radius=5,
                            fill=tuple(random.randint(70, 210) for _ in range(3)))
        d.rectangle((random.randint(300, 520), y+5, 565, y+20), fill=(55, 65, 78))
    # stand and small environmental variation
    d.rectangle((285, 405, 355, 435), fill=(35, 38, 44))
    d.polygon([(255, 450), (385, 450), (420, 465), (220, 465)], fill=(20, 22, 26))
    return im

def state_image(cls):
    im = make_base()
    # screen box in the full image (YOLO normalized xywh)
    box = (45, 55, 595, 405)
    if cls == 'black_screen':
        d = ImageDraw.Draw(im)
        d.rectangle(box, fill=(2, 2, 3))
        if random.random() < 0.35:
            d.ellipse((420, 90, 540, 180), fill=(18, 18, 20))
    elif cls == 'blurred':
        screen = im.crop(box).filter(ImageFilter.GaussianBlur(random.choice([3, 5, 8])))
        im.paste(screen, box[:2])
    elif cls == 'occluded':
        d = ImageDraw.Draw(im, 'RGBA')
        for _ in range(random.randint(1, 3)):
            x1 = random.randint(80, 480); y1 = random.randint(90, 260)
            x2 = min(580, x1 + random.randint(70, 190)); y2 = min(390, y1 + random.randint(50, 150))
            d.rounded_rectangle((x1, y1, x2, y2), radius=18,
                                fill=(random.randint(20, 80), random.randint(20, 80), random.randint(20, 80), 210))
    # small camera/lighting variation
    if random.random() < 0.35:
        im = im.filter(ImageFilter.GaussianBlur(0.5))
    return im

def yolo_line(cls_id):
    # screen rectangle: center=(320,230), size=(550,350)
    return f'{cls_id} 0.5 0.479167 0.859375 0.729167\n'

def main():
    if ROOT.exists(): shutil.rmtree(ROOT)
    for split in ['train', 'val', 'test']:
        for cls in CLASSES:
            (ROOT / 'images' / split).mkdir(parents=True, exist_ok=True)
            (ROOT / 'labels' / split).mkdir(parents=True, exist_ok=True)
    counts = {'train': 60, 'val': 10, 'test': 5}
    for cls_id, cls in enumerate(CLASSES):
        for split, n in counts.items():
            for i in range(n):
                im = state_image(cls)
                name = f'{cls}_{i:03d}.jpg'
                im.save(ROOT / 'images' / split / name, quality=92)
                (ROOT / 'labels' / split / name.replace('.jpg', '.txt')).write_text(yolo_line(cls_id), encoding='utf-8')
    abs_root = ROOT.resolve().as_posix()
    (ROOT / 'data.yaml').write_text(
        f'path: "{abs_root}"\ntrain: images/train\nval: images/val\ntest: images/test\n'
        'names: [normal, occluded, blurred, black_screen]\n', encoding='utf-8')
    (ROOT / 'README.md').write_text(
        '# Synthetic screen-status dataset\n\n'
        'Generated locally as a training baseline. Images are synthetic and must be replaced or mixed with real captures before deployment.\n', encoding='utf-8')
    print(f'created {sum(counts.values())*len(CLASSES)} images under {ROOT}')

if __name__ == '__main__': main()
