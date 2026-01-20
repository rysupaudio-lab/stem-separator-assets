# Claude Code Instructions - Stem Separator Project

## Context
Jordan (rysupaudio-lab) is building a free browser-based stem separator tool for his Shopify site rysupaudio.com. It uses Demucs AI (WebAssembly) to separate audio into drums, bass, vocals, and other stems - all client-side, no server costs.

## What's Already Done
1. ✅ Created `stem-separator.html` - Complete UI matching rysupaudio.com branding (pink/purple gradient)
2. ✅ Downloaded all 3 required files:
   - `demucs_onnx_simd.js` (75KB) - WASM loader
   - `demucs_onnx_simd.wasm` (1.1MB) - WebAssembly module
   - `ggml-model-htdemucs-4s-f16.bin` (84MB) - AI model weights
3. ✅ Created GitHub repo: https://github.com/rysupaudio-lab/stem-separator-assets
4. ✅ Uploaded the 2 smaller files via web interface

## What Needs To Be Done
1. **Upload the 84MB model file to GitHub using Git LFS** (web upload has 25MB limit)
2. **Update stem-separator.html** with the correct GitHub raw URLs in the CONFIG section
3. Optionally test that it works

## Files Location
All files are in: `/Users/jordanrys/Documents/Rys Up Audio Plugins/Stem Separator/`

## Commands to Upload Large File

```bash
cd "/Users/jordanrys/Documents/Rys Up Audio Plugins/Stem Separator"

# Install git-lfs if not installed
brew install git-lfs
git lfs install

# Clone the existing repo
git clone https://github.com/rysupaudio-lab/stem-separator-assets.git temp-repo

# Copy the large model file into the repo
cp ggml-model-htdemucs-4s-f16.bin temp-repo/

# Set up LFS tracking and push
cd temp-repo
git lfs track "*.bin"
git add .gitattributes
git add ggml-model-htdemucs-4s-f16.bin
git commit -m "Add Demucs model weights via LFS"
git push
```

## After Upload - Update HTML CONFIG

In `stem-separator.html`, find the CONFIG section (around line 612) and update to:

```javascript
const CONFIG = {
    wasmJsUrl: 'https://github.com/rysupaudio-lab/stem-separator-assets/raw/main/demucs_onnx_simd.js',
    wasmUrl: 'https://github.com/rysupaudio-lab/stem-separator-assets/raw/main/demucs_onnx_simd.wasm',
    modelUrl: 'https://github.com/rysupaudio-lab/stem-separator-assets/raw/main/ggml-model-htdemucs-4s-f16.bin',
    maxFileSize: 50 * 1024 * 1024
};
```

## Final Step
Add the HTML to Shopify: Online Store → Pages → Add page → paste HTML content
