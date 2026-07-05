// Front-end auto-reload: poll the back end's /neitz-health and reload this page when the server
// process changes (a new `boot` id). So after the back end is relaunched — e.g. the native-window
// app restarts it because viewer.py changed — the open window refreshes itself with no manual
// Cmd-R. While the server is briefly down mid-restart the fetch just fails and we keep polling;
// the next success carries the new boot id and triggers the reload. Harmless in a plain browser
// tab too (there it simply means "restart viewer.py → the tab refreshes itself").
(function () {
  var known = null;                 // boot id seen on the first successful poll
  function check() {
    fetch('/neitz-health', { cache: 'no-store' })
      .then(function (r) { return r.ok ? r.json() : Promise.reject(); })
      .then(function (h) {
        if (known === null) { known = h.boot; }
        else if (h.boot !== known) { location.reload(); }   // back end restarted → reconnect
      })
      .catch(function () { /* server restarting / unreachable — keep polling, no reload yet */ });
  }
  setInterval(check, 5000);   // gentle cadence — a restart just takes ≤5 s to reflect
  check();
})();
