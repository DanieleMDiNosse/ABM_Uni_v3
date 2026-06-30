(function () {
  "use strict";

  var RUN_ID_INPUT_ID = "stream-run-id";
  var SEQ_INPUT_ID = "stream-event-seq";
  var INITIAL_RECONNECT_MS = 400;
  var MAX_RECONNECT_MS = 10000;
  var RUN_ID_CHECK_MS = 250;
  var MIN_DISPATCH_MS = 1200;

  var source = null;
  var reconnectTimer = null;
  var reconnectDelay = INITIAL_RECONNECT_MS;
  var activeRunId = "";
  var lastSeenRunId = "";
  var seq = 0;

  function getElement(id) {
    return document.getElementById(id);
  }

  function getInputValue(id) {
    var el = getElement(id);
    if (!el) {
      return "";
    }
    return String(el.value || "").trim();
  }

  function setReactInputValue(el, value, shouldDispatch) {
    if (!el) {
      return;
    }
    var proto = Object.getPrototypeOf(el);
    var desc = proto ? Object.getOwnPropertyDescriptor(proto, "value") : null;
    if (desc && typeof desc.set === "function") {
      desc.set.call(el, value);
    } else {
      el.value = value;
    }
    if (shouldDispatch) {
      el.dispatchEvent(new Event("input", { bubbles: true }));
      el.dispatchEvent(new Event("change", { bubbles: true }));
    }
  }

  var flushTimer = null;
  var pendingDispatch = false;

  function flushDispatch() {
    flushTimer = null;
    if (!pendingDispatch) {
      return;
    }
    pendingDispatch = false;

    var seqInput = getElement(SEQ_INPUT_ID);
    if (!seqInput) {
      return;
    }

    var parsedSeq = Number(seq);
    if (!Number.isFinite(parsedSeq)) {
      parsedSeq = 0;
    }
    parsedSeq += 1;
    seq = parsedSeq;
    setReactInputValue(seqInput, String(parsedSeq), true);
  }

  function requestDispatch(immediate) {
    pendingDispatch = true;
    if (immediate) {
      if (flushTimer) {
        window.clearTimeout(flushTimer);
        flushTimer = null;
      }
      flushDispatch();
      return;
    }
    if (!flushTimer) {
      flushTimer = window.setTimeout(flushDispatch, MIN_DISPATCH_MS);
    }
  }

  function publishEvent(kind) {
    if (kind === "snapshot" || kind === "end" || kind === "stream_error" || kind === "status_change") {
      requestDispatch(true);
      return;
    }
    if (kind === "metrics_delta" || kind === "heartbeat") {
      requestDispatch(false);
    }
  }

  function closeStream() {
    if (reconnectTimer) {
      window.clearTimeout(reconnectTimer);
      reconnectTimer = null;
    }
    if (source) {
      try {
        source.close();
      } catch (_err) {
        // ignore close errors
      }
      source = null;
    }
  }

  function scheduleReconnect(runId) {
    if (!runId || runId !== activeRunId) {
      return;
    }
    if (reconnectTimer) {
      return;
    }
    var delay = reconnectDelay;
    reconnectTimer = window.setTimeout(function () {
      reconnectTimer = null;
      if (activeRunId === runId) {
        connect(runId);
      }
    }, delay);
    reconnectDelay = Math.min(MAX_RECONNECT_MS, Math.floor(reconnectDelay * 2));
  }

  function bindEvent(name) {
    if (!source) {
      return;
    }
    source.addEventListener(name, function () {
      publishEvent(name);
      if (name === "end") {
        closeStream();
      }
    });
  }

  function connect(runId) {
    closeStream();
    if (!runId) {
      return;
    }

    var url = "/stream/run/" + encodeURIComponent(runId);
    source = new EventSource(url);
    reconnectDelay = INITIAL_RECONNECT_MS;

    ["snapshot", "metrics_delta", "end", "stream_error", "status_change", "heartbeat"].forEach(bindEvent);

    source.onopen = function () {
      reconnectDelay = INITIAL_RECONNECT_MS;
    };

    source.onerror = function () {
      if (source) {
        try {
          source.close();
        } catch (_err) {
          // ignore close errors
        }
        source = null;
      }
      scheduleReconnect(runId);
    };
  }

  function syncRunId() {
    var currentRunId = getInputValue(RUN_ID_INPUT_ID);
    if (currentRunId === lastSeenRunId) {
      return;
    }
    lastSeenRunId = currentRunId;

    if (!currentRunId) {
      activeRunId = "";
      closeStream();
      return;
    }

    activeRunId = currentRunId;
    connect(currentRunId);
  }

  function init() {
    var seqInput = getElement(SEQ_INPUT_ID);
    if (seqInput) {
      var initialSeq = Number(seqInput.value || 0);
      seq = Number.isFinite(initialSeq) ? initialSeq : 0;
    }

    window.setInterval(syncRunId, RUN_ID_CHECK_MS);
    syncRunId();
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }

  window.addEventListener("beforeunload", closeStream);
})();
