/**
 * MDX-Net Processor for Browser
 * Uses ONNX Runtime Web to run MDX-Net vocal separation
 * Based on UVR (Ultimate Vocal Remover) implementation
 * 
 * Model expects input shape: [batch, 4, 3072, time_frames]
 * - 4 channels: Left Real, Left Imag, Right Real, Right Imag
 * - 3072 frequency bins (from n_fft=6144)
 */

const MDX_CONSTANTS = {
    SAMPLE_RATE: 44100,
    N_FFT: 6144,           // MDX-Net FFT size
    HOP_LENGTH: 1024,      // Hop length  
    DIM_F: 3072,           // Frequency dimension (n_fft/2)
    DIM_C: 4,              // 4 channels: L_real, L_imag, R_real, R_imag
    CHUNK_SIZE: 256,       // Time frames per chunk
    OVERLAP: 0.5           // Overlap for chunking
};

class MDXNetProcessor {
    constructor(options = {}) {
        this.ort = options.ort || null;
        this.session = null;
        this.modelPath = options.modelPath || 'https://huggingface.co/Eddycrack864/UVR5-MDX-NET-VIP-MODELS/resolve/main/UVR-MDX-NET_Main_438.onnx';
        this.onDownloadProgress = options.onDownloadProgress || (() => { });
        this.onProgress = options.onProgress || (() => { });
    }

    async loadModel(modelPath) {
        if (modelPath) this.modelPath = modelPath;

        // Download model with progress
        const response = await fetch(this.modelPath);
        const contentLength = parseInt(response.headers.get('Content-Length') || '0');
        const reader = response.body.getReader();
        const chunks = [];
        let loaded = 0;

        while (true) {
            const { done, value } = await reader.read();
            if (done) break;
            chunks.push(value);
            loaded += value.length;
            this.onDownloadProgress(loaded, contentLength);
        }

        const modelBuffer = new Uint8Array(loaded);
        let offset = 0;
        for (const chunk of chunks) {
            modelBuffer.set(chunk, offset);
            offset += chunk.length;
        }

        // Create ONNX session
        this.session = await this.ort.InferenceSession.create(modelBuffer.buffer, {
            executionProviders: ['wasm'],
            graphOptimizationLevel: 'all'
        });

        console.log('MDX-Net model loaded.');
        console.log('Input names:', this.session.inputNames);
        console.log('Output names:', this.session.outputNames);
    }

    /**
     * Separate vocals from a stereo audio signal
     * @param {Float32Array} leftChannel - Left channel audio data
     * @param {Float32Array} rightChannel - Right channel audio data  
     * @returns {Object} { vocals: { left, right }, instrumental: { left, right } }
     */
    async separate(leftChannel, rightChannel) {
        if (!this.session) throw new Error('Model not loaded');

        const { N_FFT, HOP_LENGTH, DIM_F, DIM_C, CHUNK_SIZE, OVERLAP } = MDX_CONSTANTS;
        const numSamples = leftChannel.length;

        // Compute STFT for both channels
        console.log('Computing STFT...');
        const stftLeft = this.stft(leftChannel);
        const stftRight = this.stft(rightChannel);

        const numFrames = stftLeft.numFrames;
        const chunkStep = Math.floor(CHUNK_SIZE * (1 - OVERLAP));
        const numChunks = Math.ceil((numFrames - CHUNK_SIZE) / chunkStep) + 1;

        // Output accumulators
        const vocalMaskReal = new Float32Array(numFrames * DIM_F * 2);
        const vocalMaskImag = new Float32Array(numFrames * DIM_F * 2);
        const weights = new Float32Array(numFrames);

        console.log(`Processing ${numChunks} chunks...`);

        for (let c = 0; c < numChunks; c++) {
            const startFrame = c * chunkStep;
            const endFrame = Math.min(startFrame + CHUNK_SIZE, numFrames);
            const actualChunkSize = endFrame - startFrame;

            // Prepare input tensor: [1, 4, DIM_F, CHUNK_SIZE]
            // Channel 0: Left Real
            // Channel 1: Left Imag
            // Channel 2: Right Real
            // Channel 3: Right Imag
            const inputData = new Float32Array(1 * DIM_C * DIM_F * CHUNK_SIZE);

            for (let t = 0; t < actualChunkSize; t++) {
                const frameIdx = startFrame + t;
                for (let f = 0; f < DIM_F; f++) {
                    const stftIdx = frameIdx * (N_FFT / 2 + 1) + f;

                    // Left Real (channel 0)
                    inputData[0 * DIM_F * CHUNK_SIZE + f * CHUNK_SIZE + t] = stftLeft.real[stftIdx] || 0;
                    // Left Imag (channel 1)
                    inputData[1 * DIM_F * CHUNK_SIZE + f * CHUNK_SIZE + t] = stftLeft.imag[stftIdx] || 0;
                    // Right Real (channel 2)
                    inputData[2 * DIM_F * CHUNK_SIZE + f * CHUNK_SIZE + t] = stftRight.real[stftIdx] || 0;
                    // Right Imag (channel 3)
                    inputData[3 * DIM_F * CHUNK_SIZE + f * CHUNK_SIZE + t] = stftRight.imag[stftIdx] || 0;
                }
            }

            // Run inference
            const inputTensor = new this.ort.Tensor('float32', inputData, [1, DIM_C, DIM_F, CHUNK_SIZE]);
            const feeds = { [this.session.inputNames[0]]: inputTensor };
            const results = await this.session.run(feeds);
            const outputTensor = results[this.session.outputNames[0]];
            const outputData = outputTensor.data;

            // Accumulate output with overlap-add weighting
            for (let t = 0; t < actualChunkSize; t++) {
                const frameIdx = startFrame + t;
                const windowWeight = 0.5 * (1 - Math.cos(2 * Math.PI * t / actualChunkSize));

                for (let f = 0; f < DIM_F; f++) {
                    // Output is also [1, 4, DIM_F, CHUNK_SIZE]
                    const leftRealOut = outputData[0 * DIM_F * CHUNK_SIZE + f * CHUNK_SIZE + t] || 0;
                    const leftImagOut = outputData[1 * DIM_F * CHUNK_SIZE + f * CHUNK_SIZE + t] || 0;
                    const rightRealOut = outputData[2 * DIM_F * CHUNK_SIZE + f * CHUNK_SIZE + t] || 0;
                    const rightImagOut = outputData[3 * DIM_F * CHUNK_SIZE + f * CHUNK_SIZE + t] || 0;

                    const outIdx = frameIdx * DIM_F + f;
                    vocalMaskReal[outIdx * 2] += leftRealOut * windowWeight;
                    vocalMaskImag[outIdx * 2] += leftImagOut * windowWeight;
                    vocalMaskReal[outIdx * 2 + 1] += rightRealOut * windowWeight;
                    vocalMaskImag[outIdx * 2 + 1] += rightImagOut * windowWeight;
                }
                weights[frameIdx] += windowWeight;
            }

            this.onProgress({
                progress: (c + 1) / numChunks,
                currentSegment: c + 1,
                totalSegments: numChunks
            });
        }

        // Normalize by weights
        for (let t = 0; t < numFrames; t++) {
            if (weights[t] > 0) {
                for (let f = 0; f < DIM_F; f++) {
                    const idx = t * DIM_F + f;
                    vocalMaskReal[idx * 2] /= weights[t];
                    vocalMaskImag[idx * 2] /= weights[t];
                    vocalMaskReal[idx * 2 + 1] /= weights[t];
                    vocalMaskImag[idx * 2 + 1] /= weights[t];
                }
            }
        }

        // Build vocal STFT (model outputs the separated spectrogram directly)
        const vocalStftLeft = {
            real: new Float32Array(stftLeft.real.length),
            imag: new Float32Array(stftLeft.imag.length),
            numFrames: numFrames,
            numBins: N_FFT / 2 + 1
        };
        const vocalStftRight = {
            real: new Float32Array(stftRight.real.length),
            imag: new Float32Array(stftRight.imag.length),
            numFrames: numFrames,
            numBins: N_FFT / 2 + 1
        };

        for (let t = 0; t < numFrames; t++) {
            for (let f = 0; f < DIM_F && f < vocalStftLeft.numBins; f++) {
                const stftIdx = t * vocalStftLeft.numBins + f;
                const maskIdx = t * DIM_F + f;

                vocalStftLeft.real[stftIdx] = vocalMaskReal[maskIdx * 2];
                vocalStftLeft.imag[stftIdx] = vocalMaskImag[maskIdx * 2];
                vocalStftRight.real[stftIdx] = vocalMaskReal[maskIdx * 2 + 1];
                vocalStftRight.imag[stftIdx] = vocalMaskImag[maskIdx * 2 + 1];
            }
        }

        // Compute instrumental as original - vocals
        const instStftLeft = {
            real: stftLeft.real.map((v, i) => v - vocalStftLeft.real[i]),
            imag: stftLeft.imag.map((v, i) => v - vocalStftLeft.imag[i]),
            numFrames: numFrames,
            numBins: N_FFT / 2 + 1
        };
        const instStftRight = {
            real: stftRight.real.map((v, i) => v - vocalStftRight.real[i]),
            imag: stftRight.imag.map((v, i) => v - vocalStftRight.imag[i]),
            numFrames: numFrames,
            numBins: N_FFT / 2 + 1
        };

        // Inverse STFT
        console.log('Computing inverse STFT...');
        const vocalsLeft = this.istft(vocalStftLeft, numSamples);
        const vocalsRight = this.istft(vocalStftRight, numSamples);
        const instLeft = this.istft(instStftLeft, numSamples);
        const instRight = this.istft(instStftRight, numSamples);

        return {
            vocals: { left: vocalsLeft, right: vocalsRight },
            instrumental: { left: instLeft, right: instRight }
        };
    }

    /**
     * Short-Time Fourier Transform
     */
    stft(signal) {
        const { N_FFT, HOP_LENGTH } = MDX_CONSTANTS;
        const numFrames = Math.ceil(signal.length / HOP_LENGTH);
        const numBins = N_FFT / 2 + 1;

        const real = new Float32Array(numFrames * numBins);
        const imag = new Float32Array(numFrames * numBins);

        // Hann window
        const window = new Float32Array(N_FFT);
        for (let i = 0; i < N_FFT; i++) {
            window[i] = 0.5 * (1 - Math.cos(2 * Math.PI * i / N_FFT));
        }

        for (let frame = 0; frame < numFrames; frame++) {
            const start = frame * HOP_LENGTH;

            // Extract windowed frame
            const frameData = new Float32Array(N_FFT);
            for (let i = 0; i < N_FFT; i++) {
                const idx = start + i - N_FFT / 2;
                if (idx >= 0 && idx < signal.length) {
                    frameData[i] = signal[idx] * window[i];
                }
            }

            // FFT
            const fftResult = this.fft(frameData);

            // Store results
            for (let bin = 0; bin < numBins; bin++) {
                const idx = frame * numBins + bin;
                real[idx] = fftResult.real[bin];
                imag[idx] = fftResult.imag[bin];
            }
        }

        return { real, imag, numFrames, numBins };
    }

    /**
     * Inverse Short-Time Fourier Transform
     */
    istft(stft, length) {
        const { N_FFT, HOP_LENGTH } = MDX_CONSTANTS;
        const { numFrames, numBins } = stft;

        const output = new Float32Array(length);
        const windowSum = new Float32Array(length);

        // Hann window
        const window = new Float32Array(N_FFT);
        for (let i = 0; i < N_FFT; i++) {
            window[i] = 0.5 * (1 - Math.cos(2 * Math.PI * i / N_FFT));
        }

        for (let frame = 0; frame < numFrames; frame++) {
            // Build full spectrum (mirror for negative frequencies)
            const fullReal = new Float32Array(N_FFT);
            const fullImag = new Float32Array(N_FFT);

            for (let bin = 0; bin < numBins; bin++) {
                const idx = frame * numBins + bin;
                fullReal[bin] = stft.real[idx] || 0;
                fullImag[bin] = stft.imag[idx] || 0;

                // Mirror for negative frequencies
                if (bin > 0 && bin < numBins - 1) {
                    fullReal[N_FFT - bin] = stft.real[idx] || 0;
                    fullImag[N_FFT - bin] = -(stft.imag[idx] || 0);
                }
            }

            // IFFT
            const frameData = this.ifft({ real: fullReal, imag: fullImag });

            // Overlap-add
            const start = frame * HOP_LENGTH;
            for (let i = 0; i < N_FFT; i++) {
                const idx = start + i - N_FFT / 2;
                if (idx >= 0 && idx < length) {
                    output[idx] += frameData[i] * window[i];
                    windowSum[idx] += window[i] ** 2;
                }
            }
        }

        // Normalize
        for (let i = 0; i < length; i++) {
            if (windowSum[i] > 1e-8) {
                output[i] /= windowSum[i];
            }
        }

        return output;
    }

    /**
     * FFT (Cooley-Tukey radix-2)
     */
    fft(signal) {
        const n = signal.length;
        const real = new Float32Array(n);
        const imag = new Float32Array(n);

        // Bit-reversal permutation
        for (let i = 0; i < n; i++) {
            let j = 0, bit = n >> 1;
            for (let k = i; k > 0; k >>= 1, bit >>= 1) {
                if (k & 1) j |= bit;
            }
            real[j] = signal[i];
        }

        // Cooley-Tukey FFT
        for (let size = 2; size <= n; size *= 2) {
            const halfSize = size / 2;
            const angle = -2 * Math.PI / size;

            for (let i = 0; i < n; i += size) {
                for (let j = 0; j < halfSize; j++) {
                    const theta = angle * j;
                    const cos = Math.cos(theta);
                    const sin = Math.sin(theta);

                    const re = real[i + j + halfSize] * cos - imag[i + j + halfSize] * sin;
                    const im = real[i + j + halfSize] * sin + imag[i + j + halfSize] * cos;

                    real[i + j + halfSize] = real[i + j] - re;
                    imag[i + j + halfSize] = imag[i + j] - im;
                    real[i + j] += re;
                    imag[i + j] += im;
                }
            }
        }

        return { real, imag };
    }

    /**
     * Inverse FFT
     */
    ifft(spectrum) {
        const n = spectrum.real.length;
        const real = new Float32Array(n);
        const imag = new Float32Array(n);

        // Conjugate
        for (let i = 0; i < n; i++) {
            real[i] = spectrum.real[i];
            imag[i] = -spectrum.imag[i];
        }

        // Bit-reversal
        const tempReal = new Float32Array(n);
        const tempImag = new Float32Array(n);
        for (let i = 0; i < n; i++) {
            let j = 0, bit = n >> 1;
            for (let k = i; k > 0; k >>= 1, bit >>= 1) {
                if (k & 1) j |= bit;
            }
            tempReal[j] = real[i];
            tempImag[j] = imag[i];
        }

        // Cooley-Tukey
        for (let size = 2; size <= n; size *= 2) {
            const halfSize = size / 2;
            const angle = -2 * Math.PI / size;

            for (let i = 0; i < n; i += size) {
                for (let j = 0; j < halfSize; j++) {
                    const theta = angle * j;
                    const cos = Math.cos(theta);
                    const sin = Math.sin(theta);

                    const re = tempReal[i + j + halfSize] * cos - tempImag[i + j + halfSize] * sin;
                    const im = tempReal[i + j + halfSize] * sin + tempImag[i + j + halfSize] * cos;

                    tempReal[i + j + halfSize] = tempReal[i + j] - re;
                    tempImag[i + j + halfSize] = tempImag[i + j] - im;
                    tempReal[i + j] += re;
                    tempImag[i + j] += im;
                }
            }
        }

        // Normalize
        const output = new Float32Array(n);
        for (let i = 0; i < n; i++) {
            output[i] = tempReal[i] / n;
        }

        return output;
    }
}

// Export for browser
if (typeof window !== 'undefined') {
    window.MDXNetProcessor = MDXNetProcessor;
    window.MDX_CONSTANTS = MDX_CONSTANTS;
}
