# Upload to GitHub via Command Line

Run these commands on your local machine (in the Stem Separator folder).

---

## Step 1: Install Git LFS (if not already installed)

**Mac:**
```bash
brew install git-lfs
```

**Windows:**
```bash
# Download from https://git-lfs.github.com/ and install
```

**Linux:**
```bash
sudo apt-get install git-lfs
```

Then initialize:
```bash
git lfs install
```

---

## Step 2: Create the GitHub Repository

```bash
gh repo create rysupaudio-lab/stem-separator-assets --public --description "WASM assets for Rys Up Audio stem separator"
```

Or create it manually at https://github.com/new

---

## Step 3: Initialize Local Repo & Track Large Files

```bash
cd "/path/to/Stem Separator"

# Initialize git repo
git init

# Track large files with LFS
git lfs track "*.bin"
git lfs track "*.wasm"

# Add the .gitattributes file
git add .gitattributes

# Add all files
git add demucs_onnx_simd.js
git add demucs_onnx_simd.wasm
git add ggml-model-htdemucs-4s-f16.bin

# Commit
git commit -m "Add Demucs WASM module and model weights"

# Add remote
git remote add origin https://github.com/rysupaudio-lab/stem-separator-assets.git

# Push (this will upload the large files via LFS)
git push -u origin main
```

---

## Step 4: Get Your Raw URLs

After pushing, your files will be available at:

```
https://github.com/rysupaudio-lab/stem-separator-assets/raw/main/demucs_onnx_simd.js
https://github.com/rysupaudio-lab/stem-separator-assets/raw/main/demucs_onnx_simd.wasm
https://github.com/rysupaudio-lab/stem-separator-assets/raw/main/ggml-model-htdemucs-4s-f16.bin
```

---

## Step 5: Update stem-separator.html

Replace the CONFIG section with:

```javascript
const CONFIG = {
    wasmJsUrl: 'https://github.com/rysupaudio-lab/stem-separator-assets/raw/main/demucs_onnx_simd.js',
    wasmUrl: 'https://github.com/rysupaudio-lab/stem-separator-assets/raw/main/demucs_onnx_simd.wasm',
    modelUrl: 'https://github.com/rysupaudio-lab/stem-separator-assets/raw/main/ggml-model-htdemucs-4s-f16.bin',
    maxFileSize: 50 * 1024 * 1024
};
```

---

## Quick Copy-Paste Version

If you have `gh` CLI installed and authenticated:

```bash
cd "/Users/jordanrys/Documents/Rys Up Audio Plugins/Stem Separator"
git init
git lfs install
git lfs track "*.bin" "*.wasm"
git add .gitattributes demucs_onnx_simd.js demucs_onnx_simd.wasm ggml-model-htdemucs-4s-f16.bin
git commit -m "Add Demucs WASM assets"
gh repo create rysupaudio-lab/stem-separator-assets --public --source=. --push
```

That's it! 🎉
