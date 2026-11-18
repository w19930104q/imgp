#!/usr/bin/env python3
#
# Test suite for the imgp processing pipeline stages.
#
# The matrix tests derive all 8 EXIF orientations from one fixed base
# image and verify, for each orientation x operation combination:
#   - output dimensions (orientation-normalized, then transformed)
#   - pixel hash against an independently computed reference
#   - ICC profile preservation (colour fidelity)
#   - EXIF preservation minus the consumed orientation tag
# Failure-atomicity tests verify no partial output file is left behind.
#
# Run: python3 -m unittest discover -s tests -v

import hashlib
import importlib.util
import io
import math
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from importlib.machinery import SourceFileLoader

from PIL import Image, ImageCms, ImageOps

IMGP = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'imgp'))

ORIENTATION_TAG = 0x0112
MAKE_TAG = 0x010F
MODEL_TAG = 0x0110

# display size of the 400x300 stored fixture per EXIF orientation
NORMALIZED_SIZE = {
    1: (400, 300), 2: (400, 300), 3: (400, 300), 4: (400, 300),
    5: (300, 400), 6: (300, 400), 7: (300, 400), 8: (300, 400),
}


def load_imgp():
    '''Import the imgp script as a fresh module (resets all globals)'''

    loader = SourceFileLoader('imgp_under_test', IMGP)
    spec = importlib.util.spec_from_loader(loader.name, loader)
    mod = importlib.util.module_from_spec(spec)
    loader.exec_module(mod)
    return mod


def make_icc():
    '''A real sRGB ICC profile to embed in fixtures'''

    return ImageCms.ImageCmsProfile(ImageCms.createProfile('sRGB')).tobytes()


def base_pixels(w=400, h=300):
    '''Deterministic asymmetric test pattern (orientation is detectable)'''

    img = Image.new('RGB', (w, h))
    px = img.load()
    for y in range(h):
        for x in range(w):
            px[x, y] = ((x * 7 + y) % 256, (x + y * 5) % 256, (x * 3 + y * 11) % 256)
    return img


def pixel_hash(img):
    return hashlib.sha256(img.tobytes()).hexdigest()


def jpeg_roundtrip_hash(img, quality=75):
    '''Hash of pixels after one JPEG encode/decode cycle

    imgp output is JPEG-encoded; the reference must go through the same
    single encode/decode cycle for a pixel-exact comparison.
    '''

    buf = io.BytesIO()
    img.convert('RGB').save(buf, 'JPEG', quality=quality)
    ref = Image.open(buf)
    ref.load()
    return pixel_hash(ref.convert('RGB'))


class PipelineTestCase(unittest.TestCase):

    def setUp(self):
        self.imgp = load_imgp()
        self.imgp.mute = True
        self.tmp = tempfile.mkdtemp(prefix='imgp_test_')
        self.icc = make_icc()

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def make_jpeg(self, name, orientation=1, icc=True):
        '''The one fixed fixture: same pixels, same EXIF tags, same ICC;
        only the orientation tag varies across the matrix'''

        exif = Image.Exif()
        exif[MAKE_TAG] = 'TestCam'
        exif[MODEL_TAG] = 'ModelX'
        if orientation != 1:
            exif[ORIENTATION_TAG] = orientation
        path = os.path.join(self.tmp, name)
        base_pixels().save(path, quality=95, exif=exif.tobytes(),
                           icc_profile=self.icc if icc else None)
        return path

    def make_png_rgba(self, name, orientation=1):
        img = Image.new('RGBA', (400, 300))
        px = img.load()
        for y in range(300):
            for x in range(400):
                # left half fully transparent, right half opaque
                px[x, y] = (x % 256, y % 256, 128, 0 if x < 200 else 255)
        exif = Image.Exif()
        exif[MAKE_TAG] = 'TestCam'
        exif[MODEL_TAG] = 'ModelX'
        if orientation != 1:
            exif[ORIENTATION_TAG] = orientation
        path = os.path.join(self.tmp, name)
        img.save(path, exif=exif.tobytes(), icc_profile=self.icc)
        return path

    def output_of(self, src):
        return src.replace('.jpg', '_IMGP.jpg').replace('.png', '_IMGP.png')

    def assert_metadata(self, out, expect_exif=True):
        '''ICC preserved byte-for-byte; orientation tag consumed; rest kept'''

        with Image.open(out) as img:
            self.assertEqual(hashlib.sha256(img.info['icc_profile']).hexdigest(),
                             hashlib.sha256(self.icc).hexdigest(),
                             'ICC profile not preserved')
            exif = img.getexif()
            if expect_exif:
                self.assertNotIn(ORIENTATION_TAG, exif,
                                 'orientation tag must be consumed, not written back')
                self.assertEqual(exif.get(MAKE_TAG), 'TestCam')
                self.assertEqual(exif.get(MODEL_TAG), 'ModelX')
            else:
                self.assertEqual(len(exif), 0, 'EXIF must be erased')

    # ------------------------------------------------------------- matrix

    def test_matrix_scale_50_percent(self):
        '''8 orientations x scale 50%: size, pixels, ICC, EXIF'''

        for orientation in range(1, 9):
            with self.subTest(orientation=orientation):
                src = self.make_jpeg('o%d.jpg' % orientation, orientation)
                self.imgp.scale = 50
                self.assertEqual(self.imgp.resize_image(src), [1, 0])

                out = self.output_of(src)
                nw, nh = NORMALIZED_SIZE[orientation]
                with Image.open(out) as got:
                    self.assertEqual(got.size, (nw // 2, nh // 2))
                    got.load()
                    got_rgb = got.convert('RGB')

                # independent reference: normalize, then resample
                ref = ImageOps.exif_transpose(Image.open(src))
                ref = self.imgp.gamma_aware_resize(ref, (nw // 2, nh // 2),
                                                   Image.Resampling.LANCZOS)
                self.assertEqual(pixel_hash(got_rgb),
                                 jpeg_roundtrip_hash(ref),
                                 'pixel mismatch: wrong rotate/resize order')

                self.assert_metadata(out)

    def test_matrix_fixed_hres(self):
        '''8 orientations x --res 200x0: width 200, aspect from display size'''

        for orientation in range(1, 9):
            with self.subTest(orientation=orientation):
                src = self.make_jpeg('o%d.jpg' % orientation, orientation)
                self.imgp.HRES, self.imgp.VRES = 200, 0
                self.assertEqual(self.imgp.resize_image(src), [1, 0])

                nw, nh = NORMALIZED_SIZE[orientation]
                expected = (200, self.imgp._round_half_up(nh * 200 / nw))
                with Image.open(self.output_of(src)) as got:
                    self.assertEqual(got.size, expected)
                self.assert_metadata(self.output_of(src))

    def test_matrix_rotate_90(self):
        '''8 orientations x --rotate 90: rotate normalized pixels once'''

        for orientation in range(1, 9):
            with self.subTest(orientation=orientation):
                src = self.make_jpeg('o%d.jpg' % orientation, orientation)
                self.imgp.rotate = 90
                self.assertEqual(self.imgp.rotate_image(src), [1, 0])

                out = self.output_of(src)
                ref = ImageOps.exif_transpose(Image.open(src)).rotate(-90, 0, True)
                with Image.open(out) as got:
                    self.assertEqual(got.size, ref.size)
                    got.load()
                    got_rgb = got.convert('RGB')
                self.assertEqual(pixel_hash(got_rgb),
                                 jpeg_roundtrip_hash(ref))
                self.assert_metadata(out)

    # ------------------------------------------------------- EXIF / colour

    def test_eraseexif_keeps_icc_and_normalizes(self):
        src = self.make_jpeg('o6.jpg', orientation=6)
        self.imgp.eraseexif = True
        self.imgp.scale = 50
        self.assertEqual(self.imgp.resize_image(src), [1, 0])

        out = self.output_of(src)
        with Image.open(out) as got:
            # pixels still normalized even though EXIF is erased
            self.assertEqual(got.size, (150, 200))
        self.assert_metadata(out, expect_exif=False)

    def test_png_exif_orientation(self):
        src = self.make_png_rgba('o6.png', orientation=6)
        self.imgp.scale = 50
        self.assertEqual(self.imgp.resize_image(src), [1, 0])

        out = self.output_of(src)
        with Image.open(out) as got:
            self.assertEqual(got.size, (150, 200))
            self.assertEqual(got.mode, 'RGBA')
        self.assert_metadata(out)

    # ------------------------------------------------------------- alpha

    def test_alpha_png_resize_preserves_alpha(self):
        src = self.make_png_rgba('a.png')
        self.imgp.scale = 50
        self.assertEqual(self.imgp.resize_image(src), [1, 0])

        with Image.open(self.output_of(src)) as got:
            self.assertEqual(got.mode, 'RGBA')
            self.assertEqual(got.size, (200, 150))
            alpha = got.getchannel('A')
            self.assertEqual(alpha.getextrema(), (0, 255))

    def test_alpha_png_convert_flattens(self):
        src = self.make_png_rgba('a.png')
        self.imgp.convert = True
        self.imgp.scale = 50
        self.assertEqual(self.imgp.resize_image(src), [1, 0])

        out = os.path.join(self.tmp, 'a_IMGP.jpg')
        with Image.open(out) as got:
            self.assertEqual(got.mode, 'RGB')
            self.assertEqual(got.size, (200, 150))
            # transparent half flattened onto fill_color (white)
            r, g, b = got.getpixel((10, 75))
            self.assertTrue(r > 240 and g > 240 and b > 240,
                            'alpha not flattened onto fill color')

    # ---------------------------------------------------- geometry stage

    def test_compute_target_size(self):
        m = self.imgp

        m.HRES, m.VRES = 200, 0
        self.assertEqual(m.compute_target_size(400, 300, 'JPEG'),
                         (m.RESIZE, (200, 150)))

        m.HRES, m.VRES = 0, 150
        self.assertEqual(m.compute_target_size(400, 300, 'JPEG'),
                         (m.RESIZE, (200, 150)))

        m.HRES, m.VRES = 0, 0
        m.scale = 50
        self.assertEqual(m.compute_target_size(401, 301, 'JPEG'),
                         (m.RESIZE, (201, 151)))  # half-up rounding

        m.scale = None
        m.HRES, m.VRES = 800, 600
        self.assertEqual(m.compute_target_size(400, 300, 'JPEG'),
                         (m.LOWRES, 'hres'))  # no --enlarge
        m.enlarge = True
        self.assertEqual(m.compute_target_size(400, 300, 'JPEG'),
                         (m.RESIZE, (800, 600)))
        m.enlarge = False

        m.force = True
        m.HRES, m.VRES = 100, 100
        self.assertEqual(m.compute_target_size(400, 300, 'JPEG'),
                         (m.RESIZE, (100, 100)))
        m.force = False

        m.HRES, m.VRES = 400, 300
        self.assertEqual(m.compute_target_size(400, 300, 'JPEG'),
                         (m.PASSTHROUGH, None))

        m.HRES, m.VRES = 0, 0
        m.megapixel = 0.06  # 400x300 = 0.12MP -> shrink by sqrt(0.5)
        factor = math.sqrt(0.06 / 0.12)
        self.assertEqual(m.compute_target_size(400, 300, 'JPEG'),
                         (m.RESIZE, (round(400 * factor), round(300 * factor))))
        m.megapixel = None

        m.minH, m.minV = 500, 400
        self.assertEqual(m.compute_target_size(400, 300, 'JPEG'),
                         (m.SKIP, None))
        m.minH, m.minV = 0, 0

        m.keep = True
        m.HRES, m.VRES = 400, 0
        self.assertEqual(m.compute_target_size(400, 300, 'JPEG'),
                         (m.PASSTHROUGH, None))

    # -------------------------------------------------- failure atomicity

    def test_failed_save_leaves_no_partial_file(self):
        '''RGBA cannot be encoded as JPEG: save fails, nothing is left'''

        img = Image.new('RGBA', (64, 64))
        meta = self.imgp.ImageMeta()
        meta.format = 'JPEG'
        dest = os.path.join(self.tmp, 'out.jpg')

        self.assertFalse(self.imgp.save_image(img, dest, 'JPEG', meta))
        self.assertFalse(os.path.exists(dest))
        self.assertEqual(os.listdir(self.tmp), [], 'temp file not cleaned up')

    def test_successful_save_leaves_no_temp_file(self):
        img = Image.new('RGB', (64, 64), (10, 20, 30))
        meta = self.imgp.ImageMeta()
        meta.format = 'JPEG'
        meta.icc_profile = self.icc
        dest = os.path.join(self.tmp, 'out.jpg')

        self.assertTrue(self.imgp.save_image(img, dest, 'JPEG', meta))
        self.assertEqual(os.listdir(self.tmp), ['out.jpg'])
        with Image.open(dest) as got:
            self.assertIn('icc_profile', got.info)

    def test_corrupt_source_leaves_no_output(self):
        src = os.path.join(self.tmp, 'broken.jpg')
        with open(src, 'wb') as f:
            f.write(b'\xff\xd8notajpeg' * 300)

        self.imgp.scale = 50
        self.assertIsNone(self.imgp.resize_image(src))
        self.assertEqual(os.listdir(self.tmp), ['broken.jpg'])

    def test_truncated_jpeg_leaves_no_partial_output(self):
        '''Decode failure mid-pipeline must not produce a half file'''

        src = self.make_jpeg('trunc.jpg')
        with open(src, 'r+b') as f:
            f.truncate(os.path.getsize(src) // 3)

        self.imgp.scale = 50
        try:
            self.imgp.resize_image(src)
        except OSError:
            pass  # decode error may surface from the resample stage
        self.assertEqual(os.listdir(self.tmp), ['trunc.jpg'],
                         'partial output left behind')

    # --------------------------------------------------------- CLI / e2e

    def run_cli(self, *args, cwd=None):
        return subprocess.run([sys.executable, IMGP, *args],
                              cwd=cwd or self.tmp, capture_output=True, text=True)

    def test_cli_file_mode(self):
        src = self.make_jpeg('cli.jpg', orientation=6)
        res = self.run_cli('-x', '50%', src)
        self.assertEqual(res.returncode, 0, res.stderr)
        with Image.open(self.output_of(src)) as got:
            self.assertEqual(got.size, (150, 200))
        self.assert_metadata(self.output_of(src))

    def test_cli_dir_mode_pool_workers(self):
        '''Directory traversal exercises the multiprocessing pool path'''

        for orientation in (1, 6, 8):
            self.make_jpeg('p%d.jpg' % orientation, orientation)
        res = self.run_cli('-x', '50%', self.tmp)
        self.assertEqual(res.returncode, 0, res.stderr)
        for orientation in (1, 6, 8):
            out = self.output_of(os.path.join(self.tmp, 'p%d.jpg' % orientation))
            nw, nh = NORMALIZED_SIZE[orientation]
            with Image.open(out) as got:
                self.assertEqual(got.size, (nw // 2, nh // 2))
            self.assert_metadata(out)

    def test_cli_options_kept(self):
        res = self.run_cli('--help')
        self.assertEqual(res.returncode, 0)
        for opt in ['--res', '--rotate', '--adapt', '--convert', '--eraseexif',
                    '--force', '--keep', '--minres', '--enlarge', '--nearest',
                    '--optimize', '--progressive', '--quality', '--overwrite']:
            self.assertIn(opt, res.stdout, 'CLI option lost: ' + opt)

        res = self.run_cli('-v')
        self.assertEqual(res.stdout.strip(), '3.0')


if __name__ == '__main__':
    unittest.main()
