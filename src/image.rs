//! Image preprocessing for the vision encoder, on the CPU: decode, resize to
//! a grid of 16-pixel patches whose sides are multiples of 32, normalize, and
//! lay patches out in the order the encoder merges them (2x2 blocks). Also the
//! prompt-side bookkeeping: image placeholder expansion and the 3D positions
//! the text model's rotary embedding uses for image tokens.

use crate::{Error, Result};
use half::bf16;
use image::imageops::FilterType;

pub const PATCH: usize = 16;
pub const TEMPORAL: usize = 2;
pub const MERGE: usize = 2;
/// Values per patch: 3 channels x 2 frames x 16 x 16.
pub const PATCH_VALUES: usize = 3 * TEMPORAL * PATCH * PATCH;

/// Preprocessing settings, from the model's preprocessor_config.json.
#[derive(Debug, Clone)]
pub struct Spec {
    pub min_pixels: usize,
    pub max_pixels: usize,
    pub mean: [f32; 3],
    pub std: [f32; 3],
}

impl Default for Spec {
    fn default() -> Self {
        Self {
            min_pixels: 65_536,
            max_pixels: 16_777_216,
            mean: [0.5; 3],
            std: [0.5; 3],
        }
    }
}

impl Spec {
    /// Reads preprocessor_config.json when present; `max_pixels` caps its limit.
    pub fn load(dir: &std::path::Path, max_pixels: usize) -> Result<Self> {
        let mut spec = Self::default();
        if let Ok(text) = std::fs::read_to_string(dir.join("preprocessor_config.json")) {
            let v: serde_json::Value = serde_json::from_str(&text)?;
            let get = |k: &str| v["size"][k].as_u64().map(|x| x as usize);
            spec.min_pixels = get("shortest_edge").unwrap_or(spec.min_pixels);
            spec.max_pixels = get("longest_edge").unwrap_or(spec.max_pixels);
            for (i, (m, s)) in v["image_mean"]
                .as_array()
                .into_iter()
                .flatten()
                .zip(v["image_std"].as_array().into_iter().flatten())
                .enumerate()
                .take(3)
            {
                spec.mean[i] = m.as_f64().unwrap_or(0.5) as f32;
                spec.std[i] = s.as_f64().unwrap_or(0.5) as f32;
            }
        }
        spec.max_pixels = spec.max_pixels.min(max_pixels).max(spec.min_pixels);
        Ok(spec)
    }
}

/// Rotary positions (time, height, width) of each prompt token, and the
/// offset that continues them after the prompt.
pub type Rope = (Vec<[i32; 3]>, i32);

/// A preprocessed image: patch values in bf16, `grid_h * grid_w` patches.
#[derive(Clone)]
pub struct Image {
    pub pixels: Vec<bf16>,
    pub grid_h: usize,
    pub grid_w: usize,
}

impl Image {
    pub fn patches(&self) -> usize {
        self.grid_h * self.grid_w
    }
    /// Embeddings the encoder produces, one per 2x2 block of patches.
    pub fn tokens(&self) -> usize {
        self.patches() / (MERGE * MERGE)
    }
}

/// Qwen's smart resize: sides rounded to multiples of `factor`, area within
/// [min_pixels, max_pixels], aspect ratio kept. Python's round (half to even).
pub fn smart_resize(
    height: usize,
    width: usize,
    factor: usize,
    min_pixels: usize,
    max_pixels: usize,
) -> Result<(usize, usize)> {
    let (h, w, f) = (height as f64, width as f64, factor as f64);
    if height == 0 || width == 0 || h.max(w) / h.min(w) > 200.0 {
        return Err(Error::Request(format!(
            "unsupported image size {width}x{height}"
        )));
    }
    let mut hb = ((h / f).round_ties_even() * f).max(f);
    let mut wb = ((w / f).round_ties_even() * f).max(f);
    if hb * wb > max_pixels as f64 {
        let beta = (h * w / max_pixels as f64).sqrt();
        hb = f.max((h / beta / f).floor() * f);
        wb = f.max((w / beta / f).floor() * f);
    } else if hb * wb < min_pixels as f64 {
        let beta = (min_pixels as f64 / (h * w)).sqrt();
        hb = (h * beta / f).ceil() * f;
        wb = (w * beta / f).ceil() * f;
    }
    Ok((hb as usize, wb as usize))
}

/// Decodes an encoded image (JPEG, PNG, WebP, GIF, BMP) and preprocesses it.
pub fn prepare(encoded: &[u8], spec: &Spec) -> Result<Image> {
    let img = image::load_from_memory(encoded)
        .map_err(|e| Error::Request(format!("cannot decode image: {e}")))?
        .to_rgb8();
    let (h, w) = smart_resize(
        img.height() as usize,
        img.width() as usize,
        PATCH * MERGE,
        spec.min_pixels,
        spec.max_pixels,
    )?;
    let resized = if (img.height() as usize, img.width() as usize) == (h, w) {
        img
    } else {
        image::imageops::resize(&img, w as u32, h as u32, FilterType::CatmullRom)
    };
    let raw = resized.into_raw();
    let at = |y: usize, x: usize, c: usize| {
        (raw[(y * w + x) * 3 + c] as f32 / 255.0 - spec.mean[c]) / spec.std[c]
    };
    Ok(patchify(h / PATCH, w / PATCH, at))
}

/// Lays out patches in 2x2-block order; each patch is [channel][frame][y][x]
/// with the still image repeated in both frames.
pub fn patchify(grid_h: usize, grid_w: usize, at: impl Fn(usize, usize, usize) -> f32) -> Image {
    let mut pixels = Vec::with_capacity(grid_h * grid_w * PATCH_VALUES);
    for (r, c) in patch_order(grid_h, grid_w) {
        for ch in 0..3 {
            for _frame in 0..TEMPORAL {
                for y in 0..PATCH {
                    for x in 0..PATCH {
                        pixels.push(bf16::from_f32(at(r * PATCH + y, c * PATCH + x, ch)));
                    }
                }
            }
        }
    }
    Image {
        pixels,
        grid_h,
        grid_w,
    }
}

/// (row, column) of each patch in encoder order: 2x2 blocks, row-major.
pub fn patch_order(grid_h: usize, grid_w: usize) -> impl Iterator<Item = (usize, usize)> {
    (0..grid_h / MERGE).flat_map(move |br| {
        (0..grid_w / MERGE).flat_map(move |bc| {
            (0..MERGE)
                .flat_map(move |ir| (0..MERGE).map(move |ic| (br * MERGE + ir, bc * MERGE + ic)))
        })
    })
}

/// Bilinear (align_corners) taps into the encoder's `side` x `side` learned
/// position table for each patch: four (index, weight) pairs.
pub fn position_taps(grid_h: usize, grid_w: usize, side: usize) -> (Vec<i32>, Vec<f32>) {
    let axis = |i: usize, n: usize| {
        let src = i as f32 * (side - 1) as f32 / (n.max(2) - 1) as f32;
        let f = src.floor();
        let t0 = (f as usize).min(side - 1);
        let t1 = (f as usize + 1).min(side - 1);
        let d = src - f;
        ([t0, t1], [1.0 - d, d])
    };
    let (mut idx, mut w) = (Vec::new(), Vec::new());
    for (r, c) in patch_order(grid_h, grid_w) {
        let (ht, hw) = axis(r, grid_h);
        let (wt, ww) = axis(c, grid_w);
        for a in 0..2 {
            for b in 0..2 {
                idx.push((ht[a] * side + wt[b]) as i32);
                w.push(hw[a] * ww[b]);
            }
        }
    }
    (idx, w)
}

/// Replaces each image placeholder token with one placeholder per image
/// embedding. Fails unless placeholders and images pair up.
pub fn expand_placeholders(ids: &[u32], images: &[Image], placeholder: u32) -> Result<Vec<u32>> {
    let mut out = Vec::with_capacity(ids.len() + images.iter().map(Image::tokens).sum::<usize>());
    let mut next = images.iter();
    for &t in ids {
        if t == placeholder {
            let img = next
                .next()
                .ok_or_else(|| Error::Request("more image placeholders than images".into()))?;
            out.extend(std::iter::repeat_n(placeholder, img.tokens()));
        } else {
            out.push(t);
        }
    }
    if next.next().is_some() {
        return Err(Error::Request("more images than image placeholders".into()));
    }
    Ok(out)
}

/// Rotary positions (time, height, width) of each prompt token, and the
/// offset that continues them for generated tokens. Text advances all three
/// together; an image's tokens share a time position and spread over its
/// merged grid, and the text after it resumes past the grid's longer side.
pub fn rope_positions(ids: &[u32], images: &[Image], placeholder: u32) -> Rope {
    let mut out = Vec::with_capacity(ids.len());
    let mut cur = 0i32;
    let mut next = images.iter();
    let mut i = 0;
    while i < ids.len() {
        if ids[i] == placeholder {
            let img = next.next().expect("placeholders were expanded per image");
            let (gh, gw) = ((img.grid_h / MERGE) as i32, (img.grid_w / MERGE) as i32);
            for r in 0..gh {
                for c in 0..gw {
                    out.push([cur, cur + r, cur + c]);
                }
            }
            i += img.tokens();
            cur += gh.max(gw);
        } else {
            out.push([cur; 3]);
            cur += 1;
            i += 1;
        }
    }
    let max = out
        .iter()
        .flat_map(|p| p.iter().copied())
        .max()
        .unwrap_or(-1);
    (out, max + 1 - ids.len() as i32)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn smart_resize_keeps_multiples_of_32_within_bounds() {
        assert_eq!(
            smart_resize(256, 256, 32, 65_536, 16_777_216).unwrap(),
            (256, 256)
        );
        // Too small: scaled up to at least 65,536 pixels.
        let (h, w) = smart_resize(100, 50, 32, 65_536, 16_777_216).unwrap();
        assert!(h * w >= 65_536 && h % 32 == 0 && w % 32 == 0 && h > w);
        // Too large: scaled down under the cap.
        let (h, w) = smart_resize(4000, 3000, 32, 65_536, 1_048_576).unwrap();
        assert!(h * w <= 1_048_576 && h % 32 == 0 && w % 32 == 0);
        assert!(smart_resize(10, 3000, 32, 65_536, 1 << 24).is_err());
    }

    #[test]
    fn patches_follow_merge_blocks() {
        let order: Vec<_> = patch_order(4, 4).take(6).collect();
        assert_eq!(order, vec![(0, 0), (0, 1), (1, 0), (1, 1), (0, 2), (0, 3)]);
        let img = patchify(2, 2, |y, x, c| (y * 32 + x + c) as f32);
        assert_eq!(img.pixels.len(), 4 * PATCH_VALUES);
        // Patch (0, 1) starts at pixel (0, 16); both frames repeat it.
        assert_eq!(img.pixels[PATCH_VALUES].to_f32(), 16.0);
        assert_eq!(img.pixels[PATCH_VALUES + 256].to_f32(), 16.0);
    }

    #[test]
    fn position_taps_interpolate_the_learned_grid() {
        let (idx, w) = position_taps(48, 48, 48);
        // An image the size of the table maps patches onto it exactly.
        assert_eq!(&idx[..4], &[0, 1, 48, 49]);
        assert_eq!(&w[..4], &[1.0, 0.0, 0.0, 0.0]);
        let (_, w) = position_taps(4, 6, 48);
        assert!(
            w.chunks(4)
                .all(|c| (c.iter().sum::<f32>() - 1.0).abs() < 1e-5)
        );
    }

    #[test]
    fn placeholders_expand_and_positions_span_the_grid() {
        let img = Image {
            pixels: vec![],
            grid_h: 4,
            grid_w: 6,
        }; // 2 x 3 merged tokens
        let ids = expand_placeholders(&[1, 9, 2], std::slice::from_ref(&img), 9).unwrap();
        assert_eq!(ids, vec![1, 9, 9, 9, 9, 9, 9, 2]);
        let (pos, delta) = rope_positions(&ids, std::slice::from_ref(&img), 9);
        assert_eq!(pos[0], [0, 0, 0]);
        assert_eq!(pos[1], [1, 1, 1]);
        assert_eq!(pos[6], [1, 2, 3]);
        // Text resumes after the longer side (3): position 4.
        assert_eq!(pos[7], [4, 4, 4]);
        assert_eq!(delta, 5 - 8);
        assert!(expand_placeholders(&[9, 9], &[img], 9).is_err());
    }
}
