/*! coi-serviceworker v0.1.7 - Guido Zuidhof, licensed under MIT */
let coepCredentialless = true;
if (typeof window === 'undefined') {
    self.addEventListener("install", () => self.skipWaiting());
    self.addEventListener("activate", (event) => event.waitUntil(self.clients.claim()));

    self.addEventListener("message", (ev) => {
        if (!ev.data) {
            return;
        } else if (ev.data.type === "deregister") {
            self.registration.unregister().then(() => {
                return self.clients.matchAll();
            }).then(clients => {
                clients.forEach(client => client.navigate(client.url));
            });
        }
    });

    self.addEventListener("fetch", function (event) {
        const r = event.request;
        if (r.cache === "only-if-cached" && r.mode !== "same-origin") {
            return;
        }

        const coep = coepCredentialless ? "credentialless" : "require-corp";
        const headerOverrides = {
            "cross-origin-embedder-policy": coep,
            "cross-origin-opener-policy": "same-origin",
        };

        event.respondWith(
            fetch(r).then((response) => {
                if (response.status === 0) {
                    return response;
                }

                const newHeaders = new Headers(response.headers);
                for (const key in headerOverrides) {
                    newHeaders.set(key, headerOverrides[key]);
                }

                return new Response(response.body, {
                    status: response.status,
                    statusText: response.statusText,
                    headers: newHeaders,
                });
            })
        );
    });

} else {
    (async function () {
        if (window.crossOriginIsolated === true) return;

        const registration = await navigator.serviceWorker.register(window.document.currentScript.src).catch(e => console.error("COI: ", e));
        if (registration) {
            console.log("COI: Service Worker registered. Reloading to activate headers...");
            window.location.reload();
        }
    })();
}
