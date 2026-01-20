# Rys Up Audio - Stem Separator Setup Guide

This guide shows you exactly how to download the required files and set up your GitHub repository.

---

## Files You Need (3 Total)

| File | Size | Purpose |
|------|------|---------|
| `demucs_onnx_simd.js` | ~180KB | WASM JavaScript loader |
| `demucs_onnx_simd.wasm` | ~2MB | Compiled WebAssembly module |
| `ggml-model-htdemucs-4s-f16.bin` | ~84MB | AI model weights |

---

## Step 1: Download WASM Files

### demucs_onnx_simd.js
1. Go to: https://github.com/sevagh/free-music-demixer/blob/main/web/demucs_onnx_simd.js
2. Click the **"Raw"** button (top right of the code)
3. Right-click → **"Save Page As..."** → Save as `demucs_onnx_simd.js`

### demucs_onnx_simd.wasm
1. Go to: https://github.com/sevagh/free-music-demixer/blob/main/web/demucs_onnx_simd.wasm
2. Click **"Download"** button (or View raw)
3. Save as `demucs_onnx_simd.wasm`

---

## Step 2: Download Model Weights

### ggml-model-htdemucs-4s-f16.bin (84 MB)
1. Go to: https://huggingface.co/datasets/Retrobear/demucs.cpp/tree/main
2. Find `ggml-model-htdemucs-4s-f16.bin` in the file list
3. Click the download icon (↓) on the right side
4. Wait for the 84MB download to complete

---

## Step 3: Create GitHub Repository

1. Go to https://github.com/new
2. **Owner:** rysupaudio-lab
3. **Repository name:** `stem-separator-assets`
4. **Description:** "WASM assets for Rys Up Audio stem separator"
5. **Visibility:** Public ✓
6. Click **"Create repository"**

---

## Step 4: Create a Release & Upload Files

Since GitHub limits regular file uploads to 25MB, use **Releases** for the large model file:

1. Go to your new repo: `https://github.com/rysupaudio-lab/stem-separator-assets`
2. Click **"Releases"** in the right sidebar
3. Click **"Create a new release"**
4. Fill in:
   - **Tag version:** `v1.0`
   - **Release title:** `Stem Separator WASM Assets`
   - **Description:** `Demucs WASM module and model weights for browser-based stem separation`
5. **Drag and drop** all 3 files into the "Attach binaries" area:
   - `demucs_onnx_simd.js`
   - `demucs_onnx_simd.wasm`
   - `ggml-model-htdemucs-4s-f16.bin`
6. Click **"Publish release"**

---

## Step 5: Get Your Download URLs

After publishing, your files will be available at:

```
https://github.com/rysupaudio-lab/stem-separator-assets/releases/download/v1.0/demucs_onnx_simd.js
https://github.com/rysupaudio-lab/stem-separator-assets/releases/download/v1.0/demucs_onnx_simd.wasm
https://github.com/rysupaudio-lab/stem-separator-assets/releases/download/v1.0/ggml-model-htdemucs-4s-f16.bin
```

---

## Step 6: Update stem-separator.html

Open `stem-separator.html` and find the CONFIG section (around line 612). Update it with your URLs:

```javascript
const CONFIG = {
    wasmJsUrl: 'https://github.com/rysupaudio-lab/stem-separator-assets/releases/download/v1.0/demucs_onnx_simd.js',
    wasmUrl: 'https://github.com/rysupaudio-lab/stem-separator-assets/releases/download/v1.0/demucs_onnx_simd.wasm',
    modelUrl: 'https://github.com/rysupaudio-lab/stem-separator-assets/releases/download/v1.0/ggml-model-htdemucs-4s-f16.bin',
    maxFileSize: 50 * 1024 * 1024
};
```

---

## Step 7: Add to Shopify

1. Log into your Shopify admin at rysupaudio.com
2. Go to **Online Store → Pages**
3. Click **"Add page"**
4. Title: `Free Stem Separator`
5. Click the **`<>`** button to switch to HTML mode
6. Paste the entire contents of `stem-separator.html`
7. Click **Save**
8. Go to **Online Store → Navigation** to add it to your menu

---

## Checklist

- [ ] Downloaded `demucs_onnx_simd.js`
- [ ] Downloaded `demucs_onnx_simd.wasm`
- [ ] Downloaded `ggml-model-htdemucs-4s-f16.bin` (84 MB)
- [ ] Created GitHub repo `rysupaudio-lab/stem-separator-assets`
- [ ] Created Release v1.0 with all 3 files attached
- [ ] Updated CONFIG URLs in `stem-separator.html`
- [ ] Added page to Shopify

---

## Troubleshooting

### "Failed to load WASM module"
- Make sure the GitHub release is **public**
- Check that the URLs are correct (case-sensitive!)
- Try in Chrome or Edge (best WASM support)

### Model download is slow
- The 84MB model only downloads once per session
- Users with slow connections may need to wait 1-2 minutes

### Processing fails with memory error
- Close other browser tabs
- Try a shorter audio file
- Recommend users use a desktop/laptop, not mobile

---

## Need Help?

The stem separator is based on the open-source [demucs.cpp](https://github.com/sevagh/demucs.cpp) project (MIT licensed).

For issues, check the [free-music-demixer](https://github.com/sevagh/free-music-demixer) repo for reference.
