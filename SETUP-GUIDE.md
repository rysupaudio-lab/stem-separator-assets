# Rys Up Audio - Free Stem Separator Setup Guide

This guide walks you through hosting the stem separator on your Shopify site.

---

## Overview

The stem separator has two parts:
1. **HTML/JS Interface** (stem-separator.html) - The user interface
2. **AI Model Files** - The WASM code and model weights (~82MB total)

Since Shopify doesn't allow hosting large files (>20MB), you'll need to host the AI files elsewhere (free options available).

---

## Step 1: Get the AI Model Files

You need three files from the demucs.cpp project:

### Option A: Download Pre-Built Files (Easiest)

Download from HuggingFace:
```
https://huggingface.co/datasets/sevagh/demucs.cpp-wasm/tree/main
```

You need:
- `demucs.js` (~69KB) - WASM glue code
- `demucs.wasm` (~566KB) - Compiled WebAssembly
- `ggml-model-htdemucs-4s-f16.bin` (~81MB) - Model weights

### Option B: Build Yourself (Advanced)

See: https://github.com/sevagh/demucs.cpp/tree/main/src_wasm

---

## Step 2: Host the AI Files (Free Options)

Since these files are large, host them on a CDN:

### Option 1: Cloudflare R2 (Recommended - Free)
1. Create Cloudflare account at cloudflare.com
2. Go to R2 Object Storage (free tier: 10GB storage, 10M requests/month)
3. Create a bucket named `rysup-stems`
4. Upload the 3 files
5. Enable public access and note the URLs

### Option 2: GitHub Releases (Free)
1. Create a GitHub repo (e.g., `rysupaudio/stem-separator-assets`)
2. Go to Releases → Create Release
3. Upload the 3 files as release assets
4. URLs will be like: `https://github.com/rysupaudio/stem-separator-assets/releases/download/v1.0/demucs.js`

### Option 3: Backblaze B2 (Free)
1. Create account at backblaze.com
2. Create a B2 bucket (10GB free)
3. Upload files and enable public access

---

## Step 3: Update the HTML File

Open `stem-separator.html` and find the CONFIG section near the top of the script:

```javascript
const CONFIG = {
    wasmJsUrl: 'https://your-cdn.com/demucs.js',
    wasmUrl: 'https://your-cdn.com/demucs.wasm',
    modelUrl: 'https://your-cdn.com/ggml-model-htdemucs-4s-f16.bin',
    maxFileSize: 50 * 1024 * 1024
};
```

Replace with your actual URLs:

```javascript
const CONFIG = {
    wasmJsUrl: 'https://your-r2-bucket.r2.cloudflarestorage.com/demucs.js',
    wasmUrl: 'https://your-r2-bucket.r2.cloudflarestorage.com/demucs.wasm',
    modelUrl: 'https://your-r2-bucket.r2.cloudflarestorage.com/ggml-model-htdemucs-4s-f16.bin',
    maxFileSize: 50 * 1024 * 1024
};
```

---

## Step 4: Add to Shopify

### Method 1: Custom Page (Recommended)

1. **Log into Shopify Admin**

2. **Create a new page:**
   - Go to Online Store → Pages
   - Click "Add page"
   - Title: "Free Stem Separator"
   - In the content editor, click `< >` to switch to HTML mode

3. **Paste the HTML:**
   - Copy EVERYTHING from `stem-separator.html`
   - Paste into the HTML editor

4. **Save the page**

5. **Add to navigation:**
   - Go to Online Store → Navigation
   - Edit your main menu
   - Add "Free Stem Separator" linking to your new page

### Method 2: Embedded in Existing Page

If you want it on an existing page (like a tools page):

1. Edit the page
2. Switch to HTML mode
3. Paste the code where you want it to appear

### Method 3: Custom Liquid Template (Advanced)

For a dedicated template:

1. Go to Online Store → Themes → Edit Code
2. Create new template: `page.stem-separator.liquid`
3. Paste:
```liquid
{% section 'header' %}

<div class="stem-separator-wrapper">
  <!-- Paste stem-separator.html content here -->
</div>

{% section 'footer' %}
```

4. Assign the template to your Stem Separator page

---

## Step 5: Test It!

1. Visit your page (e.g., `rysupaudio.com/pages/stem-separator`)
2. Upload a short MP3 (start with 30-60 seconds to test)
3. Click "Separate Stems"
4. Wait for processing (first load downloads the 81MB model)
5. Download your stems!

---

## Troubleshooting

### "Model failed to load"
- Check that your CDN URLs are correct
- Ensure CORS is enabled on your hosting (R2/B2 do this automatically)
- Try in Chrome/Edge (better WASM support)

### "Out of memory"
- Close other browser tabs
- Try a shorter audio file first
- Use Chrome (handles memory better than Safari)

### Processing takes forever
- This is normal! 5-30 minutes depending on:
  - Song length
  - Device CPU power
  - Available RAM
- Mobile devices will be much slower than computers

### Stems sound weird
- The 4-stem model works best on pop/rock
- Very complex mixes may have some bleed between stems
- This is the same quality as the command-line Demucs

---

## Customization

### Change Colors
Find the CSS `:root` and color values (search for `#d88aa0`, `#9c6b98`) and replace with your brand colors.

### Change File Size Limit
Edit `maxFileSize` in CONFIG (currently 50MB).

### Add More Stems
Use the 6-stem model (`ggml-model-htdemucs-6s-f16.bin`) for drums, bass, vocals, other, guitar, and piano. You'll need to update the UI to show 6 stem cards.

---

## Cost Breakdown

| Item | Cost |
|------|------|
| Shopify | Your existing plan |
| Cloudflare R2 | Free (under 10GB/10M requests) |
| Processing | Free (runs on visitor's device) |
| **Total** | **$0/month** |

---

## Support

If you run into issues, the key resources are:
- demucs.cpp GitHub: https://github.com/sevagh/demucs.cpp
- Cloudflare R2 Docs: https://developers.cloudflare.com/r2/

Good luck! 🎵
