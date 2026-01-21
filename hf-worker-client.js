/**
 * Hugging Face Gradio Client for Vocal Separation
 * Specifically for the Politrees/audio-separator_UVR space
 */

class HFWorkerClient {
    constructor(options = {}) {
        this.spaceUrl = options.spaceUrl || 'https://politrees-audio-separator-uvr.hf.space';
        this.onStatus = options.onStatus || (() => { });
        this.onProgress = options.onProgress || (() => { });
    }

    async separate(file, modelName = "MelBand Roformer Kim | Big Beta v5e FT by Unwa") {
        try {
            // 1. Upload File
            this.onStatus('Uploading audio to Hugging Face...');
            const uploadUrl = `${this.spaceUrl}/gradio_api/upload`;
            const formData = new FormData();
            formData.append('files', file);

            const uploadRes = await fetch(uploadUrl, {
                method: 'POST',
                body: formData
            });

            if (!uploadRes.ok) throw new Error(`Upload failed: ${uploadRes.statusText}`);
            const uploadData = await uploadRes.json();
            const tempFilePath = uploadData[0];
            console.log("Uploaded to HF:", tempFilePath);

            // 2. Queue Join (The modern way for long-running GPU tasks)
            const session_hash = Math.random().toString(36).substring(2, 12);

            // Replicate the exact FileData structure seen in production sniff
            const hfFileObj = {
                path: tempFilePath,
                orig_name: file.name,
                size: file.size,
                mime_type: file.type || "audio/wav",
                meta: { _type: "gradio.FileData" },
                url: `${this.spaceUrl}/gradio_api/file=${tempFilePath}`
            };

            const payload = {
                data: [
                    hfFileObj,
                    modelName,
                    256,   // Segment Size
                    false, // Override Segment Size
                    8,     // Overlap
                    0,     // Pitch Shift
                    "/tmp/PolUVR-models/",
                    "output",
                    "wav", // Output Format
                    0.9,
                    0,
                    1,
                    "NAME_(STEM)_MODEL"
                ],
                event_data: null,
                fn_index: 5,
                session_hash: session_hash,
                trigger_id: 28 // Essential: Tells the server which 'button' we clicked
            };

            this.onStatus('Joining processing queue...');

            const joinUrl = `${this.spaceUrl}/gradio_api/queue/join`;
            const joinRes = await fetch(joinUrl, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });

            if (!joinRes.ok) {
                const errorText = await joinRes.text();
                throw new Error(`Queue join failed (${joinRes.status}): ${errorText}`);
            }

            // 3. Poll for Completion via SSE
            return await this.pollQueueStatus(session_hash);

        } catch (err) {
            console.error("HF Worker Error:", err);
            throw err;
        }
    }

    async pollQueueStatus(session_hash) {
        return new Promise((resolve, reject) => {
            const statusUrl = `${this.spaceUrl}/gradio_api/queue/data?session_hash=${session_hash}`;
            const eventSource = new EventSource(statusUrl);

            eventSource.onmessage = async (event) => {
                const data = JSON.parse(event.data);
                console.log("HF Queue Event:", data.msg, data);

                switch (data.msg) {
                    case 'send_hash':
                        // Handshake msg, usually ignored if we sent it in URL
                        break;

                    case 'queue_full':
                        eventSource.close();
                        reject(new Error("Hugging Face queue is currently full. Try again in a minute."));
                        break;

                    case 'estimation':
                        if (data.rank !== undefined) {
                            this.onStatus(`Waiting in queue (Rank: ${data.rank + 1})...`);
                        }
                        break;

                    case 'progress':
                        if (data.progress_data && data.progress_data[0]) {
                            const prog = data.progress_data[0];
                            let percent = 0;

                            // Gradio 5 often provides a normalized 'progress' field (0.0 to 1.0)
                            if (typeof prog.progress === 'number') {
                                percent = prog.progress;
                            } else if (prog.index !== null && prog.total !== null && prog.total > 0) {
                                percent = prog.index / prog.total;
                            } else if (prog.index !== null && prog.length !== null && prog.length > 0) {
                                percent = prog.index / prog.length;
                            }

                            this.onProgress(percent, prog.desc || `Processing: ${prog.unit || ''}`);
                        }
                        break;

                    case 'process_generating':
                        this.onStatus('Separating stems on GPU...');
                        break;

                    case 'process_completed':
                        eventSource.close();
                        if (data.success) {
                            const results = data.output.data;
                            resolve(await this.processResults(results));
                        } else {
                            reject(new Error("HF Processing failed: " + (data.output.error || "Unknown error")));
                        }
                        break;

                    case 'heartbeat':
                        // Just keep the connection alive
                        break;
                }
            };

            eventSource.onerror = (err) => {
                console.error("SSE Error:", err);
                eventSource.close();
                reject(new Error("Connection to HF Worker lost."));
            };
        });
    }

    async processResults(data) {
        this.onStatus('Downloading results...');
        console.log("HF RAW OUTPUT:", JSON.stringify(data, null, 2));

        // Helper to find audio-like URLs/paths in any nested structure
        const findAudioUrls = (obj, acc = []) => {
            if (!obj) return acc;
            if (typeof obj === 'string') {
                if (obj.match(/\.(wav|mp3|flac|ogg)$/i) || obj.startsWith('/tmp/') || obj.includes('/file=')) {
                    acc.push(obj);
                }
            } else if (Array.isArray(obj)) {
                obj.forEach(item => findAudioUrls(item, acc));
            } else if (typeof obj === 'object') {
                if (obj.url) acc.push(obj.url);
                else if (obj.path) acc.push(obj.path);
                else if (obj.name) findAudioUrls(obj.name, acc);
                else Object.values(obj).forEach(val => findAudioUrls(val, acc));
            }
            return acc;
        };

        const allUrls = findAudioUrls(data);
        console.log("Found Audio URLs:", allUrls);

        if (allUrls.length < 2) {
            console.error("HF Output Data:", data);
            throw new Error(`Could not find 2 audio results. Found: ${allUrls.length}. Check console for 'HF RAW OUTPUT'.`);
        }

        // Assuming first is Vocals, second is Instrumental (or vice versa, typically indices map to outputs)
        // In most UVR spaces: 0=Vocals, 1=Instrumental
        const vPath = allUrls[0];
        const iPath = allUrls[1];

        const getFullUrl = (pathOrUrl) => {
            if (!pathOrUrl) return null;
            if (pathOrUrl.startsWith('http')) return pathOrUrl;
            // Clean path if it already has /file=
            const cleanPath = pathOrUrl.replace(`${this.spaceUrl}/gradio_api/file=`, '');
            return `${this.spaceUrl}/gradio_api/file=${cleanPath}`;
        };

        const vUrl = getFullUrl(vPath);
        const iUrl = getFullUrl(iPath);

        this.onStatus('Fetching audio blobs...');

        try {
            const [vBlob, iBlob] = await Promise.all([
                fetch(vUrl).then(r => r.blob()),
                fetch(iUrl).then(r => r.blob())
            ]);

            return {
                vocalsBlob: vBlob,
                instrumentalBlob: iBlob,
                vocalsUrl: URL.createObjectURL(vBlob),
                instrumentalUrl: URL.createObjectURL(iBlob)
            };
        } catch (e) {
            console.error("Blob Fetch Error:", e);
            throw new Error("Failed to download separated stems.");
        }
    }
}
