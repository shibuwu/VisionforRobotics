"""
download_textures.py — procedural + OSM map textures for the floor plane.
"""
import os
import time
import urllib.request
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

TEXTURE_DIR = os.path.join(os.path.dirname(__file__), 'textures')
os.makedirs(TEXTURE_DIR, exist_ok=True)


# --- procedural textures ---

def make_checkerboard(size=1024, squares=32):
    img = np.zeros((size, size, 3), dtype=np.uint8)
    sq = size // squares
    for i in range(squares):
        for j in range(squares):
            if (i + j) % 2 == 0:
                img[i*sq:(i+1)*sq, j*sq:(j+1)*sq] = [220, 200, 180]
            else:
                img[i*sq:(i+1)*sq, j*sq:(j+1)*sq] = [80, 60, 50]
    return Image.fromarray(img)

def make_noise(size=1024, seed=42):
    rng = np.random.default_rng(seed)
    return Image.fromarray(rng.integers(0, 256, (size, size, 3), dtype=np.uint8))

def make_tiles(size=1024, tile_size=64):
    rng = np.random.default_rng(10)
    img = np.zeros((size, size, 3), dtype=np.uint8)
    for i in range(0, size, tile_size):
        for j in range(0, size, tile_size):
            color = rng.integers(100, 220, 3)
            img[i:i+tile_size-2, j:j+tile_size-2] = color
            img[i+tile_size-2:i+tile_size, j:j+tile_size] = [40, 40, 40]
            img[i:i+tile_size, j+tile_size-2:j+tile_size] = [40, 40, 40]
    return Image.fromarray(img)

def make_concrete(size=1024):
    rng = np.random.default_rng(77)
    base = rng.integers(120, 170, (size, size, 3), dtype=np.uint8)
    pil = Image.fromarray(base).filter(ImageFilter.GaussianBlur(3))
    speckle = rng.integers(-20, 20, (size, size, 3), dtype=np.int16)
    return Image.fromarray(np.clip(np.array(pil).astype(np.int16) + speckle, 0, 255).astype(np.uint8))

def make_brick(size=1024, bw=64, bh=32):
    img = np.full((size, size, 3), [180, 80, 60], dtype=np.uint8)
    rng = np.random.default_rng(88)
    for row in range(0, size, bh):
        offset = (bw // 2) if (row // bh) % 2 else 0
        for col in range(-bw, size + bw, bw):
            c = col + offset
            color = rng.integers(140, 200), rng.integers(50, 100), rng.integers(40, 80)
            r0, r1 = max(0, row), min(size, row + bh - 2)
            c0, c1 = max(0, c), min(size, c + bw - 2)
            if r0 < r1 and c0 < c1:
                img[r0:r1, c0:c1] = color
    return Image.fromarray(img)

def make_carpet(size=1024):
    rng = np.random.default_rng(55)
    base = np.full((size, size, 3), [140, 90, 60], dtype=np.uint8)
    for _ in range(5000):
        x, y = rng.integers(0, size, 2)
        r = rng.integers(2, 8)
        color = rng.integers(80, 180, 3)
        base[max(0,y-r):min(size,y+r), max(0,x-r):min(size,x+r)] = color
    return Image.fromarray(base).filter(ImageFilter.GaussianBlur(1))

def make_wood(size=1024):
    rng = np.random.default_rng(33)
    img = np.zeros((size, size, 3), dtype=np.uint8)
    for y in range(size):
        base = np.array([160, 120, 70]) + rng.integers(-10, 10, 3)
        ring = int(15 * np.sin(y * 0.05 + rng.uniform(0, 2*np.pi)))
        img[y] = np.clip(base + ring, 0, 255)
    return Image.fromarray(img).filter(ImageFilter.GaussianBlur(1))

def make_marble(size=1024):
    rng = np.random.default_rng(99)
    base = np.full((size, size, 3), [230, 225, 220], dtype=np.uint8)
    for _ in range(200):
        x = rng.integers(0, size)
        y = rng.integers(0, size)
        length = rng.integers(50, 300)
        angle = rng.uniform(0, np.pi)
        gray = rng.integers(150, 200)
        pil = Image.fromarray(base)
        draw = ImageDraw.Draw(pil)
        x2 = x + int(length * np.cos(angle))
        y2 = y + int(length * np.sin(angle))
        draw.line([(x, y), (x2, y2)], fill=(gray, gray-5, gray-10), width=rng.integers(1, 4))
        base = np.array(pil)
    return Image.fromarray(base).filter(ImageFilter.GaussianBlur(1))

def make_grass(size=1024):
    rng = np.random.default_rng(22)
    img = np.zeros((size, size, 3), dtype=np.uint8)
    img[:,:,0] = rng.integers(40, 80, (size, size))
    img[:,:,1] = rng.integers(100, 170, (size, size))
    img[:,:,2] = rng.integers(20, 60, (size, size))
    return Image.fromarray(img).filter(ImageFilter.GaussianBlur(2))

def make_pebbles(size=1024):
    rng = np.random.default_rng(44)
    img = np.full((size, size, 3), [180, 170, 150], dtype=np.uint8)
    for _ in range(3000):
        cx, cy = rng.integers(0, size, 2)
        r = rng.integers(3, 12)
        gray = rng.integers(100, 220)
        pil = Image.fromarray(img)
        draw = ImageDraw.Draw(pil)
        draw.ellipse([cx-r, cy-r, cx+r, cy+r], fill=(gray, gray-5, gray-15))
        img = np.array(pil)
    return Image.fromarray(img)

def make_gradient(size=1024):
    img = np.zeros((size, size, 3), dtype=np.uint8)
    for y in range(size):
        for x in range(size):
            img[y, x] = [int(255*x/size), int(255*y/size), 128]
    return Image.fromarray(img)

def make_circles(size=1024):
    rng = np.random.default_rng(66)
    img = np.full((size, size, 3), [240, 235, 230], dtype=np.uint8)
    pil = Image.fromarray(img)
    draw = ImageDraw.Draw(pil)
    for _ in range(150):
        cx, cy = rng.integers(0, size, 2)
        r = rng.integers(10, 80)
        color = tuple(rng.integers(50, 220, 3))
        draw.ellipse([cx-r, cy-r, cx+r, cy+r], outline=color, width=2)
    return pil

def make_stripes(size=1024):
    rng = np.random.default_rng(11)
    img = np.zeros((size, size, 3), dtype=np.uint8)
    y = 0
    while y < size:
        h = rng.integers(8, 40)
        color = rng.integers(50, 220, 3)
        img[y:min(y+h, size)] = color
        y += h
    return Image.fromarray(img)

procedural = {
    'checkerboard.png': make_checkerboard,
    'noise.png': make_noise,
    'tiles.png': make_tiles,
    'concrete.png': make_concrete,
    'brick.png': make_brick,
    'carpet.png': make_carpet,
    'wood.png': make_wood,
    'marble.png': make_marble,
    'grass.png': make_grass,
    'pebbles.png': make_pebbles,
    'gradient.png': make_gradient,
    'circles.png': make_circles,
    'stripes.png': make_stripes,
}


# --- OSM map textures (4x4 tiles at zoom 16, stitched to 1024x1024) ---

OSM_LOCATIONS = [
    ('osm_manhattan.png',     16, 19297, 24629),   # NYC Empire State area
    ('osm_paris.png',         16, 33183, 22539),   # Paris Arc de Triomphe
    ('osm_tokyo.png',         16, 58197, 25809),   # Tokyo Shinjuku
    ('osm_london.png',        16, 32742, 21790),   # London Trafalgar Sq
    ('osm_rome.png',          16, 35037, 24350),   # Rome Piazza Venezia
    ('osm_barcelona.png',     16, 33160, 24475),   # Barcelona Eixample
    ('osm_amsterdam.png',     16, 33658, 21536),   # Amsterdam Dam Square
    ('osm_berlin.png',        16, 35201, 21491),   # Berlin Brandenburg
    ('osm_chicago.png',       16, 16814, 24354),   # Chicago Loop
    ('osm_sf.png',            16, 10480, 25329),   # San Francisco downtown
    ('osm_singapore.png',     16, 51672, 32532),   # Singapore Marina Bay
    ('osm_dubai.png',         16, 42827, 28022),   # Dubai downtown
    ('osm_moscow.png',        16, 39614, 20485),   # Moscow Red Square
    ('osm_beijing.png',       16, 53955, 24831),   # Beijing Tiananmen
    ('osm_mumbai.png',        16, 46025, 29253),   # Mumbai CST station
    ('osm_cairo.png',         16, 38452, 27027),   # Cairo Tahrir Square
    ('osm_istanbul.png',      16, 38041, 24567),   # Istanbul Sultanahmet
    ('osm_seoul.png',         16, 55881, 25376),   # Seoul Gwanghwamun
    ('osm_sydney.png',        16, 60292, 39325),   # Sydney CBD
    ('osm_mexico_city.png',   16, 14719, 29158),   # Mexico City Zocalo
    ('osm_buenos_aires.png',  16, 22137, 39487),   # Buenos Aires Obelisco
    ('osm_toronto.png',       16, 18314, 23915),   # Toronto downtown
    ('osm_bangkok.png',       16, 51061, 30237),   # Bangkok Grand Palace
    ('osm_lisbon.png',        16, 31102, 25109),   # Lisbon Baixa
    ('osm_vienna.png',        16, 35746, 22722),   # Vienna Stephansplatz
]


def download_osm_texture(name, zoom, x_start, y_start, grid=4):
    path = os.path.join(TEXTURE_DIR, name)
    if os.path.exists(path):
        return False

    canvas = Image.new('RGB', (256 * grid, 256 * grid))
    for dx in range(grid):
        for dy in range(grid):
            tx = x_start + dx
            ty = y_start + dy
            url = f'https://tile.openstreetmap.org/{zoom}/{tx}/{ty}.png'
            req = urllib.request.Request(url, headers={
                'User-Agent': 'VIO-StudentProject/1.0 (educational use)'
            })
            try:
                data = urllib.request.urlopen(req, timeout=15).read()
                import io
                tile = Image.open(io.BytesIO(data))
                canvas.paste(tile, (dx * 256, dy * 256))
            except Exception as e:
                print(f"  Failed tile {tx},{ty}: {e}")
                return False
            time.sleep(0.3)

    canvas = canvas.resize((1024, 1024), Image.LANCZOS)
    canvas.save(path)
    return True


if __name__ == '__main__':
    # generate procedural textures
    for name, gen in procedural.items():
        path = os.path.join(TEXTURE_DIR, name)
        gen().save(path)
        print(f"Created {path}")

    # download OSM city map textures
    print("\nDownloading OpenStreetMap city textures...")
    n = 0
    for name, z, x, y in OSM_LOCATIONS:
        ok = download_osm_texture(name, z, x, y)
        if ok:
            n += 1
            print(f"Downloaded {name}")
        elif os.path.exists(os.path.join(TEXTURE_DIR, name)):
            print(f"Already have {name}")
        else:
            print(f"Failed {name}")

    print(f"Downloaded {n} new OSM textures")

    total = len([f for f in os.listdir(TEXTURE_DIR) if f.endswith(('.png', '.jpg'))])
    print(f"\nTotal textures available: {total}")
