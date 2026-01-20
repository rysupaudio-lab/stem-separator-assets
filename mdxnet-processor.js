/**
 * MDX-Net Processor for Browser
 * Uses ONNX Runtime Web to run MDX-Net vocal separation
 * Based on UVR (Ultimate Vocal Remover) implementation
 */

// STFT/iSTFT implementation for MDX-Net
const MDX_CONSTANTS = {
    SAMPLE_RATE: 44100,
    N_FFT: 6144,           // MDX-Net default FFT size
    HOP_LENGTH: 1024,      // Hop length
    DIM_F: 2048,           // Frequency dimension
    DIM_T: 256,            // Time dimension (segment length)
    OVERLAP: 0.25          // Overlap for chunking
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

        console.log('MDX-Net model loaded. Input names:', this.session.inputNames, 'Output names:', this.session.outputNames);
    }

    /**
     * Separate vocals from a stereo audio signal
     * @param {Float32Array} leftChannel - Left channel audio data
     * @param {Float32Array} rightChannel - Right channel audio data  
     * @returns {Object} { vocals: { left, right }, instrumental: { left, right } }
     */
    async separate(leftChannel, rightChannel) {
        if (!this.session) throw new Error('Model not loaded');

        const numSamples = leftChannel.length;
        const segmentSamples = MDX_CONSTANTS.DIM_T * MDX_CONSTANTS.HOP_LENGTH;
        const overlap = Math.floor(segmentSamples * MDX_CONSTANTS.OVERLAP);
        const step = segmentSamples - overlap;

        // Calculate number of segments
        const numSegments = Math.ceil((numSamples - overlap) / step);

        // Output arrays
        const vocalsLeft = new Float32Array(numSamples);
        const vocalsRight = new Float32Array(numSamples);
        const instLeft = new Float32Array(numSamples);
        const instRight = new Float32Array(numSamples);
        const weights = new Float32Array(numSamples);

        for (let i = 0; i < numSegments; i++) {
            const start = i * step;
            const end = Math.min(start + segmentSamples, numSamples);

            // Extract segment
            let segmentLeft = leftChannel.slice(start, end);
            let segmentRight = rightChannel.slice(start, end);

            // Pad if necessary
            if (segmentLeft.length < segmentSamples) {
                const padding = segmentSamples - segmentLeft.length;
                const paddedLeft = new Float32Array(segmentSamples);
                const paddedRight = new Float32Array(segmentSamples);
                paddedLeft.set(segmentLeft);
                paddedRight.set(segmentRight);
                segmentLeft = paddedLeft;
                segmentRight = paddedRight;
            }

            // Compute STFT
            const stftLeft = this.stft(segmentLeft);
            const stftRight = this.stft(segmentRight);

            // Prepare input tensor [batch, channels, freq, time]
            // MDX-Net expects magnitude spectrogram
            const inputData = new Float32Array(1 * 2 * MDX_CONSTANTS.DIM_F * MDX_CONSTANTS.DIM_T);

            for (let t = 0; t < MDX_CONSTANTS.DIM_T; t++) {
                for (let f = 0; f < MDX_CONSTANTS.DIM_F; f++) {
                    const idx = t * MDX_CONSTANTS.DIM_F + f;
                    if (idx < stftLeft.magnitude.length) {
                        // Channel 0 (left)
                        inputData[0 * MDX_CONSTANTS.DIM_F * MDX_CONSTANTS.DIM_T + f * MDX_CONSTANTS.DIM_T + t] = stftLeft.magnitude[idx];
                        // Channel 1 (right)
                        inputData[1 * MDX_CONSTANTS.DIM_F * MDX_CONSTANTS.DIM_T + f * MDX_CONSTANTS.DIM_T + t] = stftRight.magnitude[idx];
                    }
                }
            }

            // Run inference
            const inputTensor = new this.ort.Tensor('float32', inputData, [1, 2, MDX_CONSTANTS.DIM_F, MDX_CONSTANTS.DIM_T]);
            const feeds = { [this.session.inputNames[0]]: inputTensor };
            const results = await this.session.run(feeds);
            const outputTensor = results[this.session.outputNames[0]];
            const outputData = outputTensor.data;

            // Extract vocal mask and apply to spectrogram
            const vocalMaskLeft = new Float32Array(stftLeft.magnitude.length);
            const vocalMaskRight = new Float32Array(stftRight.magnitude.length);

            for (let t = 0; t < MDX_CONSTANTS.DIM_T; t++) {
                for (let f = 0; f < MDX_CONSTANTS.DIM_F; f++) {
                    const idx = t * MDX_CONSTANTS.DIM_F + f;
                    if (idx < vocalMaskLeft.length) {
                        vocalMaskLeft[idx] = outputData[0 * MDX_CONSTANTS.DIM_F * MDX_CONSTANTS.DIM_T + f * MDX_CONSTANTS.DIM_T + t] || 0;
                        vocalMaskRight[idx] = outputData[1 * MDX_CONSTANTS.DIM_F * MDX_CONSTANTS.DIM_T + f * MDX_CONSTANTS.DIM_T + t] || 0;
                    }
                }
            }

            // Apply mask to get vocals
            const vocalStftLeft = {
                real: stftLeft.real.map((v, i) => v * Math.min(1, Math.max(0, vocalMaskLeft[i]))),
                imag: stftLeft.imag.map((v, i) => v * Math.min(1, Math.max(0, vocalMaskLeft[i])))
            };
            const vocalStftRight = {
                real: stftRight.real.map((v, i) => v * Math.min(1, Math.max(0, vocalMaskRight[i]))),
                imag: stftRight.imag.map((v, i) => v * Math.min(1, Math.max(0, vocalMaskRight[i])))
            };

            // Apply inverse mask to get instrumental
            const instStftLeft = {
                real: stftLeft.real.map((v, i) => v * (1 - Math.min(1, Math.max(0, vocalMaskLeft[i])))),
                imag: stftLeft.imag.map((v, i) => v * (1 - Math.min(1, Math.max(0, vocalMaskLeft[i]))))
            };
            const instStftRight = {
                real: stftRight.real.map((v, i) => v * (1 - Math.min(1, Math.max(0, vocalMaskRight[i])))),
                imag: stftRight.imag.map((v, i) => v * (1 - Math.min(1, Math.max(0, vocalMaskRight[i]))))
            };

            // Inverse STFT
            const vocalSegmentLeft = this.istft(vocalStftLeft, segmentSamples);
            const vocalSegmentRight = this.istft(vocalStftRight, segmentSamples);
            const instSegmentLeft = this.istft(instStftLeft, segmentSamples);
            const instSegmentRight = this.istft(instStftRight, segmentSamples);

            // Overlap-add with Hann window
            for (let j = 0; j < segmentSamples && start + j < numSamples; j++) {
                const windowWeight = 0.5 * (1 - Math.cos(2 * Math.PI * j / segmentSamples));
                vocalsLeft[start + j] += vocalSegmentLeft[j] * windowWeight;
                vocalsRight[start + j] += vocalSegmentRight[j] * windowWeight;
                instLeft[start + j] += instSegmentLeft[j] * windowWeight;
                instRight[start + j] += instSegmentRight[j] * windowWeight;
                weights[start + j] += windowWeight;
            }

            this.onProgress({
                progress: (i + 1) / numSegments,
                currentSegment: i + 1,
                totalSegments: numSegments
            });
        }

        // Normalize by overlap weights
        for (let i = 0; i < numSamples; i++) {
            if (weights[i] > 0) {
                vocalsLeft[i] /= weights[i];
                vocalsRight[i] /= weights[i];
                instLeft[i] /= weights[i];
                instRight[i] /= weights[i];
            }
        }

        return {
            vocals: { left: vocalsLeft, right: vocalsRight },
            instrumental: { left: instLeft, right: instRight }
        };
    }

    /**
     * Short-Time Fourier Transform
     */
    stft(signal) {
        const nFft = MDX_CONSTANTS.N_FFT;
        const hopLength = MDX_CONSTANTS.HOP_LENGTH;
        const numFrames = Math.ceil(signal.length / hopLength);
        const numBins = nFft / 2 + 1;

        const real = new Float32Array(numFrames * numBins);
        const imag = new Float32Array(numFrames * numBins);
        const magnitude = new Float32Array(numFrames * numBins);

        // Hann window
        const window = new Float32Array(nFft);
        for (let i = 0; i < nFft; i++) {
            window[i] = 0.5 * (1 - Math.cos(2 * Math.PI * i / nFft));
        }

        for (let frame = 0; frame < numFrames; frame++) {
            const start = frame * hopLength;

            // Extract windowed frame
            const frameData = new Float32Array(nFft);
            for (let i = 0; i < nFft; i++) {
                const idx = start + i - nFft / 2;
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
                magnitude[idx] = Math.sqrt(fftResult.real[bin] ** 2 + fftResult.imag[bin] ** 2);
            }
        }

        return { real, imag, magnitude, numFrames, numBins };
    }

    /**
     * Inverse Short-Time Fourier Transform
     */
    istft(stft, length) {
        const nFft = MDX_CONSTANTS.N_FFT;
        const hopLength = MDX_CONSTANTS.HOP_LENGTH;
        const numFrames = Math.ceil(length / hopLength);
        const numBins = nFft / 2 + 1;

        const output = new Float32Array(length);
        const windowSum = new Float32Array(length);

        // Hann window
        const window = new Float32Array(nFft);
        for (let i = 0; i < nFft; i++) {
            window[i] = 0.5 * (1 - Math.cos(2 * Math.PI * i / nFft));
        }

        for (let frame = 0; frame < numFrames; frame++) {
            // Build full spectrum (mirror for negative frequencies)
            const fullReal = new Float32Array(nFft);
            const fullImag = new Float32Array(nFft);

            for (let bin = 0; bin < numBins; bin++) {
                const idx = frame * numBins + bin;
                fullReal[bin] = stft.real[idx] || 0;
                fullImag[bin] = stft.imag[idx] || 0;

                // Mirror for negative frequencies
                if (bin > 0 && bin < numBins - 1) {
                    fullReal[nFft - bin] = stft.real[idx] || 0;
                    fullImag[nFft - bin] = -(stft.imag[idx] || 0);
                }
            }

            // IFFT
            const frameData = this.ifft({ real: fullReal, imag: fullImag });

            // Overlap-add
            const start = frame * hopLength;
            for (let i = 0; i < nFft; i++) {
                const idx = start + i - nFft / 2;
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
     * Simple FFT implementation (Cooley-Tukey radix-2)
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

        // FFT
        const tempSignal = new Float32Array(n);
        for (let i = 0; i < n; i++) {
            tempSignal[i] = real[i];
        }

        // Bit-reversal
        for (let i = 0; i < n; i++) {
            let j = 0, bit = n >> 1;
            for (let k = i; k > 0; k >>= 1, bit >>= 1) {
                if (k & 1) j |= bit;
            }
            real[j] = tempSignal[i];
            imag[j] = -spectrum.imag[i];
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

                    const re = real[i + j + halfSize] * cos - imag[i + j + halfSize] * sin;
                    const im = real[i + j + halfSize] * sin + imag[i + j + halfSize] * cos;

                    real[i + j + halfSize] = real[i + j] - re;
                    imag[i + j + halfSize] = imag[i + j] - im;
                    real[i + j] += re;
                    imag[i + j] += im;
                }
            }
        }

        // Conjugate and normalize
        const output = new Float32Array(n);
        for (let i = 0; i < n; i++) {
            output[i] = real[i] / n;
        }

        return output;
    }
}

// Export for browser
if (typeof window !== 'undefined') {
    window.MDXNetProcessor = MDXNetProcessor;
    window.MDX_CONSTANTS = MDX_CONSTANTS;
}
