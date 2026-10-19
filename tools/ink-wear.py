# Generates a seamless print-wear mask for the monogram (CSS mask uses the alpha channel):
# opaque = ink, transparent = where the paper shows through.
# Usage: python3 tools/ink-wear.py out.webp [amount]  -> then base64 it into --wear
# amount sets how worn the print is: 1.35 for v2, 0.8 for the lighter v4.
import numpy as np
from PIL import Image
import sys
N = 1536
rng = np.random.default_rng(23)
fx = np.fft.fftfreq(N)[:, None]; fy = np.fft.fftfreq(N)[None, :]
def field(sigma):
    a = rng.standard_normal((N, N))
    g = np.exp(-2 * (np.pi * sigma) ** 2 * (fx ** 2 + fy ** 2))
    b = np.real(np.fft.ifft2(np.fft.fft2(a) * g))
    return (b - b.mean()) / b.std()
broad = field(90)          # where the press was lighter or heavier
patch = field(14)          # mid-size thin patches
tooth = field(0.9)         # paper fibres / pinholes
wear = np.clip(0.5 + 0.5 * broad, 0, 1) ** 1.6     # 0 clean .. 1 worn
# pinholes: more of them where the print is worn
amount = float(sys.argv[2]) if len(sys.argv) > 2 else 1.35
thr = -3.0 - 0.3 * (1.35 - amount) + amount * wear + 0.35 * amount / 1.35 * np.clip(-patch, 0, None)
alpha = np.clip(1 + (tooth - thr) * 1.8, 0, 1)
# faint overall tooth: ink a touch thinner on the paper's high points
alpha = np.round(alpha * 7) / 7   # few levels keeps the file small
a = Image.fromarray((alpha * 255).astype(np.uint8), 'L')
black = Image.new('L', a.size, 0)
Image.merge('RGBA', [black, black, black, a]).save(sys.argv[1], 'WEBP', lossless=True, quality=100)
print('holes %.2f%%  partial %.2f%%' % ((alpha < .2).mean() * 100, ((alpha >= .2) & (alpha < .9)).mean() * 100))
